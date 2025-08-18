# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""



import pickle
import socket
import threading
import ray
import zmq
import csv
import os
import time

import numpy as np
import torch
import torch.distributed

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from packaging import version as vs
from typing import Any, Dict, List, Union
from zoneinfo import ZoneInfo


from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from types import MethodType

from loguru import logger
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.model_executor.sampling_metadata import SamplingMetadata

from siirl import DataProto
from siirl.utils.debug import GPUMemoryLogger
from siirl.utils.model_utils.torch_functional import get_response_mask, pad_2d_list_to_length
from siirl.utils.params import RolloutArguments
from siirl.workers.rollout.base import BaseRollout
from siirl.utils.extras.device import is_cuda_available

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for siirl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutArguments, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: RolloutArguments
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        # micro_batch_size for mini-batch inference
        self.micro_batch_size = config.micro_batch_size if config.micro_batch_size else 0
        if self.micro_batch_size > 0:
            logger.info(f"Mini-batch inference is enabled with micro_batch_size: {self.micro_batch_size}")

        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.tensor_model_parallel_size
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.max_num_batched_tokens

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = deepcopy(config.engine_kwargs.vllm)
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.limit_images:  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        vllm_version = version("vllm")
        if vs.parse(vllm_version) >= vs.parse("0.9.0") and is_cuda_available:
            logger.info(f"Add environment variable for vLLM version {vllm_version}")
            # This environment variable is mandatory due to an issue where PyTorch 2.7.0
            #   causes flashinfer to fail with the error `FlashInfer requires sm75+`.
            # This can be removed once a later version of PyTorch or flashinfer fixes the issue.
            #   Reference: https://github.com/flashinfer-ai/flashinfer/issues/1101
            cap = torch.cuda.get_device_capability(torch.cuda.current_device())
            os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}+PTX"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.seed,  # Use None for random seed to avoid identical outputs for repeated inputs
            **lora_kwargs,
            **engine_kwargs,
        )


        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        dictConfig = config.to_dict()
        for k in dictConfig.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = dictConfig.get(k)

        # kwargs["n"] = 1  # already repeat in ray_trainer

        logger.info(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        if "internvl" in model_hf_config.model_type:
            stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
            stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
            self.sampling_params.stop_token_ids = stop_token_ids
            if "internlm2" in model_hf_config.llm_config.model_type:
                logger.info("Set vllm tokenizer for internlm2")
                self.inference_engine.set_tokenizer(tokenizer)

        self.pad_token_id = tokenizer.pad_token_id

        self.enbale_perf = os.environ.get("SIIRL_ENABLE_PERF", "0") == "1"
        if self.enbale_perf:
            self.perf_step = 1
            world_size = torch.distributed.get_world_size()
            model_name = os.path.basename(os.path.normpath(model_path))
            ts = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d-%H")
            self.perf_log_path = os.path.join("performance_logs", model_name, ts)
            os.makedirs(self.perf_log_path, exist_ok=True)
            self.perf_log_file = os.path.join(self.perf_log_path, f"rollout_world_size_{world_size}.csv")
            with open(self.perf_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "rank", "timestamp", "batch_size", "inference_latency_s", "min_prompt_len", "max_prompt_len", "avg_prompt_len", "min_gen_len", "max_gen_len", "avg_gen_len"])

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for i, input_data in enumerate(vllm_inputs):
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")


        prompt_lengths = []
        if self.enbale_perf:
            # Record prompt length
            prompt_lengths = [len(item["prompt_token_ids"]) for item in vllm_inputs]
            torch.cuda.synchronize()
            start_time = time.time()

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            logger.info(f"vllm generate sampling params: {self.sampling_params}")
            # if micro_batch_size is configured, split the batch into smaller chunks
            # and generate sequences for each chunk sequentially.
            if self.micro_batch_size > 0:
                outputs = []
                for i in range(0, len(vllm_inputs), self.micro_batch_size):
                    micro_batch = vllm_inputs[i : i + self.micro_batch_size]
                    if not micro_batch:
                        continue

                    micro_outputs = self.inference_engine.generate(
                        prompts=micro_batch,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                    outputs.extend(micro_outputs)
            else:
                # full-batch inference
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

            if self.enbale_perf:
                rank = torch.distributed.get_rank()
                torch.cuda.synchronize()
                inference_latency = time.time() - start_time
                # Record the length of generated tokens
                generated_lengths = [len(o.outputs[0].token_ids) for o in outputs]
                # Compute statistics
                min_prompt_len = np.min(prompt_lengths) if prompt_lengths else 0
                max_prompt_len = np.max(prompt_lengths) if prompt_lengths else 0
                avg_prompt_len = np.mean(prompt_lengths) if prompt_lengths else 0

                min_gen_len = np.min(generated_lengths) if generated_lengths else 0
                max_gen_len = np.max(generated_lengths) if generated_lengths else 0
                avg_gen_len = np.mean(generated_lengths) if generated_lengths else 0
                # Write to CSV file
                with open(self.perf_log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.perf_step, rank, time.strftime("%Y-%m-%d %H:%M:%S"), batch_size, inference_latency, min_prompt_len, max_prompt_len, avg_prompt_len, min_gen_len, max_gen_len, avg_gen_len])
                self.perf_step += 1
                logger.info(f"vllm rollout perf log saved to {self.perf_log_file}")

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            meta_info = {"total_input_tokens": 0, "total_output_tokens": 0}
            for input_data in vllm_inputs:
                meta_info["total_input_tokens"] += len(input_data["prompt_token_ids"])

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)
                meta_info["total_output_tokens"] += len(output.outputs[0].token_ids)
            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(non_tensor_batch["multi_modal_inputs"], self.sampling_params.n)
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(
                        non_tensor_batch["interaction_kwargs"], self.sampling_params.n
                    )
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt"], self.sampling_params.n
                    )

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                'rollout_log_probs': rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)



# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)

class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/siirl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/siirl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()
        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)


    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))
    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
