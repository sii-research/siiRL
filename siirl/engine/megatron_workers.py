# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, Infrawaves. All rights reserved.
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
The main entry point to run the PPO algorithm
"""

import os
import time
import warnings
from typing import Union
import datetime
import psutil

import torch
import torch.distributed
from codetiming import Timer
from loguru import logger

from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, NonTensorData
try:
    from mindspeed.megatron_adaptor import repatch
except ImportError:
    repatch = None

from megatron.core import parallel_state as mpu

from siirl.engine.base_worker.megatron.worker import MegatronWorker
from siirl.models.loader import load_tokenizer
from siirl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from siirl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from siirl.utils.model_utils.flops_counter import FlopsCounter
from siirl.utils.extras.fs import copy_to_local
from siirl.utils.megatron.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from siirl.utils.extras.import_utils import import_external_libs
from siirl.utils.extras.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device
from siirl.utils.model_utils.model import get_hf_model_path, load_mcore_dist_weights, load_megatron_gptmodel_weights
from siirl.utils.model_utils.torch_dtypes import PrecisionType
from siirl.params.model_args import ActorRolloutRefArguments
from siirl.engine.actor.megatron_actor import MegatronPPOActor
from siirl.engine.critic.megatron_critic import MegatronPPOCritic
from siirl.engine.reward_model.megatron.reward_model import MegatronRewardModel


def set_random_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# TODO(Ping Zhang): We will deprecate this hybrid worker in the future.
class ActorRolloutRefWorker(MegatronWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, process_group=None):
        super().__init__()
        self.config = config
        global_mindspeed_repatch(self.config.megatron.override_transformer_config)
        # self.process_group = process_group

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            # Use LOCAL_RANK for device setting, but respect process group for distributed ops
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend=get_nccl_backend())
            get_torch_device().set_device(rank)
            if self.config.actor.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.actor.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.actor.megatron.seed)

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # normalize config
        if self._is_actor and self._is_rollout:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            if self.config.actor.ppo_micro_batch_size:
                self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

            self._is_offload_param = self.config.actor.megatron.param_offload
            self._is_offload_grad = self.config.actor.megatron.grad_offload
            self._is_offload_optimizer = self.config.actor.megatron.optimizer_offload
        elif self._is_ref:
            if self.config.ref.log_prob_micro_batch_size:
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
            else:
                assert self.config.ref.log_prob_micro_batch_size_per_gpu is not None, "Please note that in the ref policy configuration, `log_prob_micro_batch_size_per_gpu` and `log_prob_micro_batch_size` should not be None at the same time."
            self._ref_is_offload_param = self.config.ref.megatron.param_offload

    def _build_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config):
        from megatron.core.models.gpt.gpt_model import ModelType

        from siirl.utils.megatron.optimizer import get_megatron_optimizer
        from siirl.utils.megatron.megatron_utils import get_model, init_megatron_optim_config
        from siirl.utils.model_utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(model_path, model_path, self.dtype, override_model_config, override_transformer_config, self.config.model.trust_remote_code)
        self.generation_config = get_generation_config(self.local_path)

        def megatron_actor_model_provider(pre_process, post_process):
            from siirl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(self.tf_config, self.hf_config, pre_process, post_process, share_embeddings_and_output_weights=self.share_embeddings_and_output_weights, value=False, freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False))
            parallel_model.to(get_device_name())
            return parallel_model

        actor_module = None
        # Step 3: initialize the megatron model
        if self._is_actor and self._is_rollout:
            actor_module = get_model(
                megatron_actor_model_provider,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            print(f"actor_module: {len(actor_module)}")
            if self.config.actor.load_weight:
                if self.config.actor.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(actor_module, self.config.actor.megatron.dist_checkpointing_path, is_value_model=False)
                else:
                    load_megatron_gptmodel_weights(self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False)

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
        elif self._is_ref:
            print(f"self.config.ref.load_weight: {self.config.ref.load_weight}")
            ref_module = get_model(
                model_provider_func=megatron_actor_model_provider,
                model_type=ModelType.encoder_or_decoder,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )
            # ref_module = nn.ModuleList(ref_module)

            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                print("load ref weight start")
                if self.config.ref.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(ref_module, self.config.ref.megatron.dist_checkpointing_path, is_value_model=False)
                else:
                    load_megatron_gptmodel_weights(self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False)
            log_gpu_memory_usage("After ref module init", logger=logger)
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config = init_megatron_optim_config(optim_config)
            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config)
        else:
            optim_config = None
            actor_optimizer = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return actor_module, actor_optimizer, self.hf_config, optim_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        if self.config.rollout.name == "vllm":
            from torch.distributed.device_mesh import init_device_mesh

            from siirl.engine.rollout.vllm_rollout import vllm_mode, vLLMRollout
            # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
            # we will reorganize their weight format when resharding from actor to rollout.

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            rollout_device_mesh = init_device_mesh(get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
            log_gpu_memory_usage("Before building vllm rollout", logger=None)

            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.use_shm)
            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                )
            elif vllm_mode == "spmd":
                rollout = vLLMRollout(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                )
            log_gpu_memory_usage("After building vllm rollout", logger=logger)

        elif self.config.rollout.name in ["sglang", "sglang_async"]:
            if self.config.rollout.name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            from siirl.engine.rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to SGLang's model_runner would check CUDA device capability.
            # However, due to siirl's setting, the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from siirl.engine.sharding_manager.megatron_sglang import MegatronSGLangShardingManager

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            rollout_device_mesh = init_device_mesh("cpu", mesh_shape=(dp, infer_tp, 1), mesh_dim_names=("dp", "tp", "pp"))

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=None)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=None)
        else:
            raise NotImplementedError("Only vllmRollout and SGLangRollout are supported with Megatron now")
        
        print("rollout init done")
        return rollout, None

    def init_model(self):
        import_external_libs(self.config.model.external_lib)

        override_model_config = self.config.model.override_config
        if self._is_actor:
            override_transformer_config = self.config.actor.megatron.override_transformer_config
        elif self._is_ref:
            override_transformer_config = self.config.ref.megatron.override_transformer_config
        else:
            override_transformer_config = None
        
        if not override_transformer_config:
            override_transformer_config = OmegaConf.create()
        
        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)

        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            optim_config = self.config.actor.optim if self._is_actor else None
            self.actor_module, self.actor_optimizer, self.actor_model_config, self.actor_optim_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            self.actor = MegatronPPOActor(
                config=self.config.actor,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
            )
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout(trust_remote_code=self.config.model.trust_remote_code)
            # used for sleep/wake_up
            self.rollout.sharding_manager = self.sharding_manager
            log_gpu_memory_usage("After rollout init", logger=logger)

        if self._is_ref:
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After ref model init", logger=logger)
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage("After offload ref params during init", logger=logger)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                model_config=self.actor_model_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                tokenizer=self.tokenizer,
                optimizer=self.actor_optimizer,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    @GPUMemoryLogger(role="update_actor", logger=logger)
    def update_actor(self, data: TensorDict):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
        data = data.to(get_device_name())

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(data=data)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

        # TODO: here, we should return all metrics
        output = TensorDict(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        get_torch_device().empty_cache()
        return output

    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    def generate_sequences(self, prompts: TensorDict):
        assert self._is_rollout
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params during generate_sequences", logger=logger)
        prompts.batch = prompts.batch.to(get_device_name())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

        with self.sharding_manager:
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After entering sharding manager", logger=logger)

            # (zhangchi.usc1992) wake up kv cache here. Currently only support vllm.
            # Will support sglang once separate wakeup of model weights and kv cache is supported
            # This API should be exposed by the rollout. Will rewrite this part when we refactor after v0.4 release.
            # Currently, we hack here to support running large models (QWen3-236b and DeepSeek-671b)
            if self.config.rollout.name == "vllm":
                import inspect

                if "tags" in inspect.signature(self.rollout.inference_engine.wake_up).parameters:
                    self.rollout.inference_engine.wake_up(tags=["kv_cache"])

            output = self.rollout.generate_sequences(prompts=prompts)

        output = output.to("cpu")
        # clear kv cache
        get_torch_device().empty_cache()
        return output

    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.load_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.save_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)

# TODO(Ping Zhang): We will deprecate this hybrid worker in the future.
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}")
        return self.rollout.execute_method(method, *args, **kwargs)

    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    async def wake_up(self):
        await self.rollout.wake_up()
        # return something to block the caller
        return True

    async def sleep(self):
        await self.rollout.sleep()
        # return something to block the caller
        return True


class CriticWorker(MegatronWorker):
    def __init__(self, config, process_group=None):
        super().__init__()
        self.config = config
        # self.process_group = process_group

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        global_mindspeed_repatch(self.config.megatron.override_transformer_config)
        if not torch.distributed.is_initialized():
            # Use LOCAL_RANK for device setting, but respect process group for distributed ops
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend=get_nccl_backend())
            get_torch_device().set_device(rank)

            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.ppo_micro_batch_size:
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        # TODO(sgm): support critic model offload

    def _build_critic_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config):
        from siirl.utils.megatron.optimizer import get_megatron_optimizer, get_megatron_optimizer_param_scheduler
        from siirl.utils.megatron.megatron_utils import init_megatron_optim_config
        from siirl.utils.model_utils.model import print_model_size
        from siirl.utils.megatron.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        self._init_hf_config_and_tf_config(
            model_path, 
            model_path, 
            self.dtype, 
            override_model_config, 
            override_transformer_config, 
            self.config.model.trust_remote_code,
            self.config.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,  # critic is value model
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        critic_module = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
        )

        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            t0 = time.time()
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    critic_module, self.config.megatron.dist_checkpointing_path, is_value_model=True
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(critic_module, local_model_path)
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, critic_module, params_dtype=self.dtype, is_value_model=True
                    )
            t1 = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"critic load_weight time: {t1 - t0}")
        if self.rank == 0:
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        optim_config_megatron = init_megatron_optim_config(optim_config)
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
        critic_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=critic_optimizer, config=optim_config
        )
        get_torch_device().empty_cache()
        return critic_module, critic_optimizer, critic_optimizer_scheduler, self.hf_config, optim_config

    def init_model(self):
        # create critic
        import_external_libs(self.config.model.external_lib)
        override_model_config = self.config.model.override_config
        override_transformer_config = self.config.megatron.override_transformer_config

        if not override_transformer_config:
            override_transformer_config = OmegaConf.create()
        
        override_ddp_config = self.config.megatron.override_ddp_config
        if not override_ddp_config:
            override_ddp_config = OmegaConf.create()
        
        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        self.critic_module, self.critic_optimizer, self.critic_optimizer_scheduler, self.critic_model_config, critic_optimizer_config = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.checkpoint,
            model_config=self.critic_model_config,
            transformer_config=self.tf_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.critic_optimizer,
            optimizer_scheduler=self.critic_optimizer_scheduler,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            use_dist_checkpointing=self.config.megatron.use_dist_checkpointing,
        )

    def compute_values(self, data: TensorDict):
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        data["max_token_len"] = NonTensorData(self.config.forward_max_token_len_per_gpu)
        data["use_dynamic_bsz"] = NonTensorData(self.config.use_dynamic_bsz)
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        values = self.critic.compute_values(data=data)
        data["values"] = values
        data = data.to("cpu")
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        return data

    def update_critic(self, data: TensorDict):
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.critic_optimizer)
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self.critic.update_critic(data=data)
        delta_time = timer.last
        global_num_tokens = data["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops
        metrics["perf/delta_time/critic"] = delta_time
        data["metrics"] = NonTensorData(metrics)
        data = data.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)
        
        return data

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)


class RewardModelWorker(MegatronWorker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config, process_group=None):
        super().__init__()
        self.config = config
        # self.process_group = process_group

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        global_mindspeed_repatch(self.config.actor.megatron.get("override_transformer_config", {}))
        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend=get_nccl_backend())
            get_torch_device().set_device(rank)
            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_rm_model(self, model_path, tokenizer, override_model_config, override_transformer_config):
        from siirl.utils.megatron.megatron_utils import McoreModuleWrapperConfig, make_megatron_module

        self._init_hf_config_and_tf_config(
            model_path,
            tokenizer,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.trust_remote_code,
            self.config.megatron.use_mbridge,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,  # reward model is value model
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        reward_model = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            override_model_config=override_model_config,
        )

        if self.config.load_weight:
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(reward_model, self.config.megatron.dist_checkpointing_path, is_value_model=True)
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(reward_model, local_model_path)
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, reward_model, params_dtype=self.dtype, is_value_model=True
                    )

        get_torch_device().empty_cache()
        return reward_model, self.hf_config

    def init_model(self):
        # create critic
        import_external_libs(self.config.model.external_lib)
        override_model_config = self.config.model.override_config
        override_transformer_config = self.config.model.override_transformer_config

        if not override_transformer_config:
            override_transformer_config = OmegaConf.create()

        use_shm = self.config.model.use_shm
        sft_tokenizer_local_path = copy_to_local(self.config.model.input_tokenizer, use_shm=use_shm)
        sft_tokenizer = load_tokenizer(path=sft_tokenizer_local_path)["tokenizer"]
        rm_tokenizer_path = self.config.model.rm_tokenizer
        rm_tokenizer = None
        if rm_tokenizer_path is not None:
            rm_tokenizer_local_path = copy_to_local(rm_tokenizer_path, use_shm=use_shm)
            rm_tokenizer = load_tokenizer(path=rm_tokenizer_local_path)["tokenizer"]

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        reward_model_module, reward_model_config = self._build_rm_model(
            model_path=self.config.model.path,
            tokenizer=rm_tokenizer,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        # FIXME(sgm): reward model param offload is implemented in MegatronRewardModel
        # should be implemented in workers
        self.rm = MegatronRewardModel(
            config=self.config,
            reward_model_module=reward_model_module,
            model_config=reward_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            sft_tokenizer=sft_tokenizer,
            rm_tokenizer=rm_tokenizer,
        )

    # TODO: reward model use itself tokenizer instead of sft tokenizer
    # the input_ids, responses, attention_mask and position_ids may be different!
    def compute_rm_score(self, data: TensorDict):
        data.meta_info["micro_batch_size"] = self.config.micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        output = self.rm.compute_reward(data)
        output = output.to("cpu")
        return output


# ================================= Separated Workers =================================

IS_ACTOR_ROLLOUT_REF_INITIALIZED = False

def global_initialize_model_parallel(config: ActorRolloutRefArguments):
    # For separated workers, we use actor's megatron config for distributed model initialization
    megatron_config = config.actor.megatron
    
    rank = int(os.environ["LOCAL_RANK"])
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=get_nccl_backend(),
            timeout=datetime.timedelta(seconds=600),
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
    get_torch_device().set_device(rank)

    global IS_ACTOR_ROLLOUT_REF_INITIALIZED
    if IS_ACTOR_ROLLOUT_REF_INITIALIZED:
        return

    if megatron_config.sequence_parallel:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    
    mpu.initialize_model_parallel(
            tensor_model_parallel_size=megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=megatron_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=megatron_config.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=megatron_config.context_parallel_size,
            expert_model_parallel_size=megatron_config.expert_model_parallel_size,
            expert_tensor_parallel_size=megatron_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )    
    set_random_seed(seed=megatron_config.seed)
    
    IS_ACTOR_ROLLOUT_REF_INITIALIZED = True


IS_MINDSPEED_REPATCH = False

def global_mindspeed_repatch(config):
    """
    Use for Mindspeed repatch global once
    """

    global IS_MINDSPEED_REPATCH
    if repatch is not None and not IS_MINDSPEED_REPATCH:
        # NPU MindSpeed patch, will be refactored with MindSpeedEngine.
        repatch(config)
        IS_MINDSPEED_REPATCH = True


class ActorWorker(MegatronWorker):
    """
    Dedicated worker for actor training
    """

    def __init__(self, config: DictConfig, process_group=None):
        # For backward compatibility, we do not seperate the hybrid configurations,
        # i.e., the `config` here is still ActorRolloutRefArguments
        assert isinstance(config, ActorRolloutRefArguments), "config of ActorWorker must be ActorRolloutRefArguments"
        super().__init__()
        self.config = config
        global_mindspeed_repatch(self.config.actor.megatron.to_dict().get("override_transformer_config", {}))
        global_initialize_model_parallel(self.config)

        self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
        self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.actor.ppo_micro_batch_size:
            self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

        self._is_offload_param = self.config.actor.megatron.param_offload
        self._is_offload_grad = self.config.actor.megatron.grad_offload
        self._is_offload_optimizer = self.config.actor.megatron.optimizer_offload

    def _build_actor_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config):
        from siirl.utils.megatron.megatron_utils import init_megatron_optim_config
        from siirl.utils.model_utils.model import print_model_size
        from siirl.utils.megatron.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from siirl.utils.megatron.optimizer import get_megatron_optimizer, get_megatron_optimizer_param_scheduler

        self._init_hf_config_and_tf_config(
            model_path,
            model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.trust_remote_code,
            self.config.actor.megatron.use_mbridge,
        )
        wrap_config = McoreModuleWrapperConfig(
            is_value_model=False,  # actor is not value model
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
        )
        actor_module = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
        )
        print(f"actor_module: {len(actor_module)}")
        if self.config.actor.load_weight:
            if self.config.actor.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    actor_module, self.config.actor.megatron.dist_checkpointing_path, is_value_model=False
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(actor_module, local_model_path)
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False
                    )

        if self.rank == 0:
            print_model_size(actor_module[0])
        log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        # TODO: add more optimizer args into config
        optim_megatron_config = init_megatron_optim_config(optim_config)
        actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_megatron_config)
        actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
                optimizer=actor_optimizer, config=optim_config
            )

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config, optim_config

    def init_model(self):
        import_external_libs(self.config.model.external_lib)

        override_model_config = self.config.model.override_config
        override_transformer_config = self.config.actor.megatron.override_transformer_config
        
        if not override_transformer_config:
            override_transformer_config = OmegaConf.create()
        
        override_ddp_config = self.config.actor.megatron.override_ddp_config
        if not override_ddp_config:
            override_ddp_config = OmegaConf.create()
        
        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)

        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        # we need the model for actor
        optim_config = self.config.actor.optim
        self.actor_module, self.actor_optimizer, self.actor_optimizer_scheduler, self.actor_model_config, self.actor_optim_config = self._build_actor_model_optimizer(
            model_path=self.config.model.path,
            optim_config=optim_config,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        self.actor = MegatronPPOActor(
            config=self.config.actor,
            model_config=self.actor_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.actor_optimizer,
        )
        log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        self.flops_counter = FlopsCounter(self.actor_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.actor.checkpoint,
            model_config=self.actor_model_config,
            transformer_config=self.tf_config,
            role="actor",
            model=self.actor_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.actor_optimizer,
            optimizer_scheduler=self.actor_optimizer_scheduler,
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
        )
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    @GPUMemoryLogger(role="update_actor", logger=logger)
    def update_actor(self, data: TensorDict):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
        data = data.to(get_device_name())

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(data=data)
        delta_time = timer.last
        global_num_tokens = data["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops / promised_flops
        metrics["perf/delta_time/actor"] = delta_time
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        # TODO: here, we should return all metrics
        data["metrics"] = NonTensorData(metrics)
        data = data.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return data

    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    def compute_log_prob(self, data: TensorDict):
        torch.cuda.synchronize()
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params during compute_log_prob", logger=logger)

        # we should always recompute old_log_probs when it is HybridEngine
        data["micro_batch_size"] = NonTensorData(self.config.rollout.log_prob_micro_batch_size_per_gpu)
        data["max_token_len"] = NonTensorData(self.config.rollout.log_prob_max_token_len_per_gpu)
        data["use_dynamic_bsz"] = NonTensorData(self.config.rollout.log_prob_use_dynamic_bsz)
        data["temperature"] = NonTensorData(self.config.rollout.temperature)
        data = data.to(get_device_id())
        with Timer(name="compute_log_prob", logger=None) as timer:
            output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
        delta_time = timer.last

        # store results of Actor old_log_probs
        data["old_log_probs"] = output
        data["entropys"] = entropys

        # update metrics
        metrics = {}
        global_num_tokens = data["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor_log_prob"] = estimated_flops / promised_flops
        metrics["perf/delta_time/actor_log_prob"] = delta_time
        data["metrics"] = NonTensorData(metrics)
        data = data.to("cpu")
        # clear kv cache
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during compute_log_prob", logger=logger)
        return data

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)


class RolloutWorker(MegatronWorker):
    """
    Dedicated worker for rollout inference
    """

    def __init__(self, config: DictConfig, process_group=None):
        # For backward compatibility, we do not seperate the hybrid configurations,
        # i.e., the `config` here is still ActorRolloutRefArguments
        assert isinstance(config, ActorRolloutRefArguments), "config of RolloutWorker must be ActorRolloutRefArguments"
        super().__init__()
        self.config = config
        global_mindspeed_repatch(self.config.actor.megatron.to_dict().get("override_transformer_config", {}))

        # normalize rollout config
        global_initialize_model_parallel(self.config)

        if self.config.rollout.log_prob_micro_batch_size:
            self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        
        self.device_mesh = None

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"

        if self.config.rollout.name == "vllm":
            from siirl.engine.rollout.vllm_rollout import vLLMRollout
            # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
            # we will reorganize their weight format when resharding from actor to rollout.
            rollout_device_mesh = init_device_mesh(get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
            self.device_mesh = rollout_device_mesh
            log_gpu_memory_usage("Before building vllm rollout", logger=None)
            # local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.use_shm)
            rollout = vLLMRollout(
                model_path=self.local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.hf_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage("After building vllm rollout", logger=logger)

        elif self.config.rollout.name in ["sglang", "sglang_async"]:
            from siirl.engine.rollout.sglang_rollout import SGLangRollout
            if self.config.rollout.name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            rollout_device_mesh = init_device_mesh("cpu", mesh_shape=(dp, infer_tp, 1), mesh_dim_names=("dp", "tp", "pp"))
            self.device_mesh = rollout_device_mesh
            # local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=None)
            rollout = SGLangRollout(
                actor_module=self.local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.hf_config,
                trust_remote_code=trust_remote_code,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=None)
        else:
            raise NotImplementedError("Only vllmRollout and SGLangRollout are supported with Megatron now")
        
        return rollout, None

    def init_model(self):
        import_external_libs(self.config.model.external_lib)
        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init rollout inference engine", logger=logger)

        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        # Initialize HF config and tokenizer for inference engine setup
        from siirl.utils.model_utils.model import get_generation_config
        
        # self.local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.use_shm)
        # self.tokenizer = load_tokenizer(model_args=self.config.model)['tokenizer']
        model_path = self.config.model.path
        model_override_config = self.config.model.override_config
        model_override_transformer_config = self.config.actor.megatron.override_transformer_config
        model_trust_remote_code = self.config.model.trust_remote_code
        
        self._init_hf_config_and_tf_config(
            model_path, model_path, 
            self.dtype, 
            model_override_config, 
            model_override_transformer_config, 
            model_trust_remote_code, 
            False, # mbridge is not used for rollout
        )
        
        self.generation_config = get_generation_config(self.local_path)

        # Only build the inference engine (vLLM/SGLang) - no need for Megatron model
        self.rollout, self.sharding_manager = self._build_rollout(trust_remote_code=self.config.model.trust_remote_code)
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After rollout init", logger=logger)

    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    def generate_sequences(self, prompts: TensorDict):
        prompts = prompts.to(get_device_id())
        prompts["eos_token_id"] = NonTensorData(self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id)
        prompts["pad_token_id"] = NonTensorData(self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id)

        with self.sharding_manager:
            log_gpu_memory_usage("After entering sharding manager", logger=logger)
            with Timer(name="generate_sequences", logger=None) as timer:
                output = self.rollout.generate_sequences(prompts=prompts)
            delta_time = timer.last
            # Note: Add metrics for Rollout, we may use them later.
            metrics = {}
            metrics["perf/delta_time/rollout"] = delta_time
        log_gpu_memory_usage("After rollout generation", logger=logger)
        output["metrics"] = NonTensorData(metrics, batch_size=None)
        output = output.to("cpu")
        # clear kv cache
        get_torch_device().empty_cache()
        return output
    
    def set_rollout_sharding_manager(self, sharding_manager):
        self.sharding_manager = sharding_manager


class ReferenceWorker(MegatronWorker):
    """
    Dedicated worker for reference policy
    """

    def __init__(self, config: DictConfig, process_group=None):
        # For backward compatibility, we do not seperate the hybrid configurations,
        # i.e., the `config` here is still ActorRolloutRefArguments
        assert isinstance(config, ActorRolloutRefArguments), "config must be ActorRolloutRefArguments"
        super().__init__()
        self.config = config
        global_mindspeed_repatch(self.config.actor.megatron.to_dict().get("override_transformer_config", {}))
        global_initialize_model_parallel(self.config)

        # normalize ref config
        if self.config.ref.log_prob_micro_batch_size:
            self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
        else:
            assert self.config.ref.log_prob_micro_batch_size_per_gpu is not None, "Please note that in the ref policy configuration, `log_prob_micro_batch_size_per_gpu` and `log_prob_micro_batch_size` should not be None at the same time."
        
        self._ref_is_offload_param = self.config.ref.megatron.param_offload

    def _build_ref_model(self, model_path, override_model_config, override_transformer_config):
        from siirl.utils.megatron.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        self._init_hf_config_and_tf_config(
            model_path,
            model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.trust_remote_code,
            self.config.actor.megatron.use_mbridge,
        )
        wrap_config = McoreModuleWrapperConfig(
            is_value_model=False,  # ref is not value model
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
        )
        ref_module = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            override_model_config=override_model_config,
        )
        if self.config.ref.load_weight:  # should align with the actor:
            assert self.config.actor.load_weight == self.config.ref.load_weight
            print("load ref weight start")
            if self.config.ref.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    ref_module, self.config.ref.megatron.dist_checkpointing_path, is_value_model=False
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    self.bridge.load_weights(ref_module, local_model_path)
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False
                    )
        log_gpu_memory_usage("After ref module init", logger=logger)
        return ref_module, self.hf_config

    def init_model(self):
        import_external_libs(self.config.model.external_lib)

        override_model_config = self.config.model.override_config
        override_transformer_config = self.config.ref.megatron.override_transformer_config
        
        if not override_transformer_config:
            override_transformer_config = OmegaConf.create()
        
        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        log_gpu_memory_usage("Before init ref model", logger=logger)
        self.ref_module, self.ref_model_config = self._build_ref_model(
            model_path=self.config.model.path,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        log_gpu_memory_usage("After init ref model", logger=logger)
        self.ref_policy = MegatronPPOActor(
            config=self.config.ref,
            model_config=self.ref_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.ref_module,
            actor_optimizer=None,
        )
        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)
            log_gpu_memory_usage("After offload ref params during init", logger=logger)

        self.flops_counter = FlopsCounter(self.ref_model_config)
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After finish ref model init", logger=logger)

    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    def compute_ref_log_prob(self, data: TensorDict):
        if self._ref_is_offload_param:
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)
            log_gpu_memory_usage("After load ref params and grad during compute_ref_log_prob", logger=logger)
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        data["max_token_len"] = NonTensorData(self.config.ref.log_prob_max_token_len_per_gpu)
        data["use_dynamic_bsz"] = NonTensorData(self.config.ref.log_prob_use_dynamic_bsz)
        data["temperature"] = NonTensorData(self.config.rollout.temperature)
        data = data.to(get_device_id())

        with Timer(name="compute_ref_log_prob", logger=None) as timer:
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        delta_time = timer.last

        # update metrics
        metrics = {}
        global_num_tokens = data["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/ref"] = estimated_flops / promised_flops
        metrics["perf/delta_time/ref"] = delta_time
        data["metrics"] = NonTensorData(metrics)
        data["ref_log_prob"] = output
        data = data.to("cpu")
        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)
            log_gpu_memory_usage("After offload ref params and grad during compute_ref_log_prob", logger=logger)
        return data


class AsyncRolloutWorker(RolloutWorker):
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}")
        return self.rollout.execute_method(method, *args, **kwargs)

    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    async def wake_up(self):
        await self.rollout.wake_up()
        # return something to block the caller
        return True

    async def sleep(self):
        await self.rollout.sleep()
        # return something to block the caller
        return True

    def set_rollout_sharding_manager(self, sharding_manager):
        super().set_rollout_sharding_manager(sharding_manager)
        self.rollout.sharding_manager = sharding_manager
