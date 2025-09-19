# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file contains a Megatron style Hybrid Engine that shares the weights of the actor with the inference engine.
"""

import asyncio

from sglang.srt.entrypoints.engine import Engine
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from siirl.workers.databuffer.protocol import  all_gather_data_proto
from siirl import DataProto
from siirl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from siirl.utils.extras.device import get_device_id, get_device_name, get_torch_device, set_expandable_segments
from siirl.utils.megatron.megatron_utils import (
    per_tensor_generator,
    load_megatron_model_to_gpu,
    offload_megatron_model_to_cpu,
)
from siirl.utils.memory_utils import aggressive_empty_cache

from siirl.workers.sharding_manager.base import BaseShardingManager
from loguru import logger


"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor

class MultiAgentMegatronSGLangShardingManager(BaseShardingManager):
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: Engine,
        model_config,
        rollout_config,
        transformer_config,
        layer_name_mapping,
        weight_converter,
        device_mesh: DeviceMesh | None = None,
        offload_param: bool = False,
        bridge=None,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.bridge = bridge

        if self.device_mesh is not None:
            self.infer_tp_size = self.device_mesh["infer_tp"].size()
        else:
            self.infer_tp_size = self.inference_engine._tp_size

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="MultiAgentMegatronSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.wake_up())

    @GPUMemoryLogger(role="MultiAgentMegatronSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.sleep())

    async def update_weights(self, params):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39
        """
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            await self.inference_engine.resume_memory_occupation()
        named_tensors = params

        load_format = None
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor))

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                gathered_serialized_tensors = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
            else:
                gathered_serialized_tensors = None
            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                group=self.device_mesh["infer_tp"].get_group(),
            )

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                await self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=False,
                )
                
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                await self.inference_engine.flush_cache()

    async def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            await self.inference_engine.release_memory_occupation()

    @GPUMemoryLogger(role="MultiAgentMegatronSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        aggressive_empty_cache(force_sync=True)

        if self.offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
        if self.bridge is not None:
            per_tensor_param = self.bridge.export_weights(self.actor_module)
        else:
            per_tensor_param = per_tensor_generator(
                self.actor_module,
                self.model_config,
                self.weight_converter,
                self.transformer_config,
                self.layer_name_mapping,
            )

        set_expandable_segments(False)

        await self.update_weights(per_tensor_param)
        if self.offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        aggressive_empty_cache(force_sync=True)
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="MultiAgentMegatronSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        if self.rollout_config.free_cache_engine:
            log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
            await self.release_memory()
            log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        for model in self.actor_module:
            model.train()
        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
