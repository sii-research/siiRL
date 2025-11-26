# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
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

import json
import os
import warnings
from dataclasses import asdict
from typing import Union, Optional

import psutil
import torch
import torch.distributed
from codetiming import Timer
from loguru import logger
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed import ProcessGroup, init_device_mesh
from torch.distributed.device_mesh import DeviceMesh
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
import siirl.utils.model_utils.torch_functional as F
from typing import Any, Dict, List, Optional, Union, Set
from siirl.models.loader import load_tokenizer
from siirl.engine.base_worker import Worker
from siirl.execution.scheduler.enums import Role
from siirl.execution.scheduler.enums import Role
from siirl.utils.model_utils.activation_offload import enable_activation_offloading
from siirl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from siirl.utils.extras.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device, is_cuda_available, is_npu_available
from siirl.utils.model_utils.flops_counter import FlopsCounter
from siirl.utils.extras.fs import copy_to_local
from siirl.utils.model_utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from siirl.utils.extras.import_utils import import_external_libs
from siirl.utils.model_utils.model import compute_position_id_with_mask
from siirl.utils.extras.py_functional import convert_to_regular_types
from siirl.params.model_args import ActorRolloutRefArguments, CriticArguments, FSDPArguments, OptimizerArguments, RewardModelArguments
from siirl.engine.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

device_name = get_device_name()


def create_device_mesh_from_group(
    process_group: ProcessGroup,
    fsdp_size: int = 1,
    sp_size: int = 1,
) -> DeviceMesh:
    """Creates a DeviceMesh from a process group for specific parallel strategies.

    This function configures a DeviceMesh based on the provided parallelism sizes.
    It supports three mutually exclusive parallelism configurations:
    1.  Data Parallelism + Sequence Parallelism ([dp, sp]): Activated when `sp_size` > 1.
    2.  Fully Sharded Data Parallelism ([fsdp]): Activated when `fsdp_size` is 1.
    3.  Distributed Data Parallelism + FSDP ([ddp, fsdp]): Activated when `fsdp_size` > 1.

    Note:
        `sp_size > 1` and `fsdp_size > 1` cannot be used simultaneously.

    Args:
        process_group (ProcessGroup): The base process group for the mesh.
        fsdp_size (int): The size of the FSDP dimension. Activates FSDP modes.
                         Defaults to 1.
        sp_size (int): The size of the Sequence Parallel dimension. Activates [dp, sp]
                       mode if > 1. Defaults to 1.

    Returns:
        DeviceMesh: A configured DeviceMesh object for the specified topology.

    Raises:
        ValueError: If inputs are invalid, parallelism strategies are mixed,
                    or the group size is not compatible with the requested
                    parallelism dimensions.
    """
    if process_group is None:
        raise ValueError("`process_group` cannot be None.")

    if sp_size > 1 and fsdp_size > 1:
        raise ValueError("Sequence Parallelism (sp_size > 1) and FSDP (fsdp_size > 1) are mutually exclusive and cannot be activated simultaneously.")

    import torch.distributed

    device_type = get_device_name()
    group_size = torch.distributed.get_world_size(group=process_group)
    ranks_in_group = torch.distributed.get_process_group_ranks(process_group)

    # --- 2. [dp, sp] Mode ---
    if sp_size > 1:
        if group_size % sp_size != 0:
            raise ValueError(f"For [dp, sp] mode, the process group size ({group_size}) must be divisible by sp_size ({sp_size}).")
        dp_size = group_size // sp_size
        mesh_shape = (dp_size, sp_size)
        mesh_dim_names = ("dp", "sp")
        logger.info(f"Creating [dp, sp] DeviceMesh with shape {mesh_shape}.")

        rank_mesh = torch.tensor(ranks_in_group, dtype=torch.long).view(mesh_shape)
        return DeviceMesh(device_type, rank_mesh, mesh_dim_names=mesh_dim_names)

    # --- 3. FSDP / DDP Modes ---
    if fsdp_size < 0 or fsdp_size >= group_size:
        # Pure FSDP (equivalent to DDP over the whole group).
        # This creates a 1D mesh representing a single shard group over all ranks.
        logger.info("Creating pure [fsdp] DeviceMesh from the process group.")
        # mesh_tensor = torch.tensor(ranks_in_group)
        return DeviceMesh.from_group(group=process_group, device_type=device_type, mesh=ranks_in_group, mesh_dim_names=("fsdp",))

    # [ddp, fsdp] mode for fsdp_size > 1
    if group_size % fsdp_size != 0:
        raise ValueError(f"The process group size ({group_size}) must be divisible by fsdp_size ({fsdp_size}).")

    # [ddp, fsdp] mode for fsdp_size > 1
    ddp_size = group_size // fsdp_size
    mesh_shape = (ddp_size, fsdp_size)
    mesh_dim_names = ("ddp", "fsdp")
    logger.info(f"Creating [ddp, fsdp] DeviceMesh with shape {mesh_shape}.")

    rank_mesh = torch.tensor(ranks_in_group, dtype=torch.long).view(mesh_shape)
    # TODO: support 2D process group(List)
    return DeviceMesh(device_type=device_type, mesh=rank_mesh, mesh_dim_names=mesh_dim_names)


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: ActorRolloutRefArguments, role: str, process_group: ProcessGroup):
        super().__init__()
        self.config = config

        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}", rank=rank, world_size=world_size)

        self.group_world_size = torch.distributed.get_world_size(group=process_group)
        # build device mesh for FSDP
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh_from_group(process_group=process_group, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = create_device_mesh_from_group(process_group=process_group, sp_size=self.ulysses_sequence_parallel_size)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.lora_rank
        self._is_lora = self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.param_offload
            self._is_offload_optimizer = self.config.actor.fsdp_config.optimizer_offload
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.param_offload

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, \
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by " \
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, \
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than " \
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path: str,
        fsdp_config: FSDPArguments,
        optim_config: Optional[OptimizerArguments],
        override_model_config: DictConfig,
        use_remove_padding: bool = False,
        use_fused_kernels: bool = False,
        enable_gradient_checkpointing: bool = False,
        trust_remote_code: bool = False,
        use_liger: bool = False,
        role: Role = Role.Actor,
        enable_activation_offload: bool = False,
    ):
        """
        Build model and optimizer (Refactored version).
        
        This method orchestrates the model building process through 4 main steps:
            1. Prepare and load model from pretrained checkpoint
            2. Apply model modifications (LoRA, gradient checkpointing, etc.)
            3. Wrap model with FSDP
            4. Create optimizer and learning rate scheduler (Actor only)
        
        Args:
            model_path: Path to model checkpoint
            fsdp_config: FSDP configuration
            optim_config: Optimizer configuration (optional)
            override_model_config: Config overrides
            use_remove_padding: Whether to use remove padding
            use_fused_kernels: Whether to use fused kernels
            enable_gradient_checkpointing: Whether to enable gradient checkpointing
            trust_remote_code: Whether to trust remote code
            use_liger: Whether to apply Liger kernel
            role: Role (Actor or RefPolicy)
            enable_activation_offload: Whether to enable activation offload
        
        Returns:
            Tuple of (model_fsdp, optimizer, lr_scheduler, model_config)
        """
        from siirl.utils.model_utils.model import print_model_size
        
        assert role in [Role.Actor, Role.RefPolicy]
        
        # Step 1: Prepare and load model
        actor_module, actor_model_config, torch_dtype = self._prepare_and_load_model(
            model_path=model_path,
            fsdp_config=fsdp_config,
            override_model_config=override_model_config,
            trust_remote_code=trust_remote_code,
            role=role,
        )
        
        # Step 2: Apply model modifications
        actor_module = self._apply_model_modifications(
            model=actor_module,
            use_liger=use_liger,
            use_remove_padding=use_remove_padding,
            use_fused_kernels=use_fused_kernels,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            torch_dtype=torch_dtype,
        )
        
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(actor_module)
        
        # Step 3: Wrap model with FSDP
        actor_module_fsdp = self._setup_fsdp_wrapper(
            model=actor_module,
            fsdp_config=fsdp_config,
            role=role,
            enable_activation_offload=enable_activation_offload,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )
        
        # Step 4: Create optimizer and scheduler
        actor_optimizer, actor_lr_scheduler = self._create_optimizer_and_scheduler(
            model=actor_module_fsdp,
            optim_config=optim_config,
            role=role,
        )
        
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    # ==================================================================================
    # Legacy implementation (kept for rollback, remove after verification)
    # ==================================================================================
    def _build_model_optimizer_legacy(
        self,
        model_path: str,
        fsdp_config: FSDPArguments,
        optim_config: Optional[OptimizerArguments],
        override_model_config: DictConfig,
        use_remove_padding: bool = False,
        use_fused_kernels: bool = False,
        enable_gradient_checkpointing: bool = False,
        trust_remote_code: bool = False,
        use_liger: bool = False,
        role: Role = Role.Actor,
        enable_activation_offload: bool = False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoImageProcessor, AutoProcessor

        from siirl.utils.model_utils.model import get_generation_config, print_model_size, update_model_config
        from siirl.utils.model_utils.torch_dtypes import PrecisionType

        assert role in [Role.Actor, Role.RefPolicy]

        local_path = model_path

        if self.config.model.model_type == "embodied":
            if self.config.embodied.embodied_type == "openvla-oft":
                from siirl.models.embodied.openvla_oft.configuration_prismatic import OpenVLAConfig
                from siirl.models.embodied.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
                from siirl.models.embodied.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
                
                AutoConfig.register("openvla", OpenVLAConfig)
                AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
                AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
                AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
                if self.rank == 0:
                    try:
                        from siirl.utils.embodied.openvla_utils import update_auto_map, check_model_logic_mismatch
                        logger.info(f"[rank-{self.rank}] Updating auto_map for OpenVLA-OFT at {local_path}")
                        update_auto_map(local_path)
                        check_model_logic_mismatch(local_path)
                        logger.info(f"[rank-{self.rank}] Successfully updated auto_map for OpenVLA-OFT")
                    except Exception as e:
                        logger.error(f"[rank-{self.rank}] Failed to update auto_map for OpenVLA-OFT: {e}")
                        raise
                torch.distributed.barrier()
            elif self.config.embodied.embodied_type == "openvla":
                from siirl.models.embodied.openvla.configuration_prismatic import OpenVLAConfig
                from siirl.models.embodied.openvla.modeling_prismatic import OpenVLAForActionPrediction
                from siirl.models.embodied.openvla.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
                
                AutoConfig.register("openvla", OpenVLAConfig)
                AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
                AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
                AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
                if self.rank == 0:
                    try:
                        from siirl.utils.embodied.openvla_utils import update_auto_map, check_model_logic_mismatch
                        logger.info(f"[rank-{self.rank}] Updating auto_map for OpenVLA at {local_path}")
                        update_auto_map(local_path)
                        check_model_logic_mismatch(local_path)
                        logger.info(f"[rank-{self.rank}] Successfully updated auto_map for OpenVLA")
                    except Exception as e:
                        logger.error(f"[rank-{self.rank}] Failed to update auto_map for OpenVLA: {e}")
                        raise
                torch.distributed.barrier()
            else:
                raise ValueError(f"Invalid vla type: {self.config.embodied.embodied_type}")

        torch_dtype = fsdp_config.model_dtype
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        if self.config.model.model_type == "embodied" and self.config.embodied.embodied_type == "openvla-oft":
            actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        else:
            actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
        if self._is_ref:
            self.flops_counter = FlopsCounter(actor_model_config, forward_only=True)
        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        # if self.rank == 0:
        #     logger.info(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            is_embodied_model = self.config.model.model_type == "embodied"
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys() or is_embodied_model:
                actor_module_class = AutoModelForVision2Seq
            elif type(actor_model_config).__name__ in "configuration_internvl_chat.InternVLChatConfig":
                from siirl.models.transformers.internvl_chat import InternVLChatModel

                actor_module_class = InternVLChatModel
                logger.info("Choose InternVLChatModel for internvl")
            else:
                actor_module_class = AutoModelForCausalLM

            
            
            if is_embodied_model and self.config.embodied.embodied_type == "openvla-oft":
                # OpenVLA-OFT: No flash_attention_2, requires additional setup
                logger.info("Loading OpenVLA-OFT model (without flash_attention_2)")
                actor_module = actor_module_class.from_pretrained(
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    config=actor_model_config,
                    trust_remote_code=trust_remote_code,
                )
                
                # Set the number of images in input for multi-camera support
                if hasattr(actor_module, 'vision_backbone'):
                    num_images = getattr(self.config.embodied, 'num_images_in_input', 1)
                    actor_module.vision_backbone.set_num_images_in_input(num_images)
                    logger.info(f"Set vision_backbone.num_images_in_input = {num_images}")
                
                # Load dataset statistics for action normalization
                dataset_statistics_path = os.path.join(local_path, "dataset_statistics.json")
                if os.path.isfile(dataset_statistics_path):
                    with open(dataset_statistics_path, "r") as f:
                        norm_stats = json.load(f)
                    actor_module.norm_stats = norm_stats
                    logger.info(f"Loaded dataset_statistics.json with {len(norm_stats)} task(s)")
                else:
                    logger.warning(
                        "WARNING: No dataset_statistics.json file found for OpenVLA-OFT checkpoint.\n"
                        "You can ignore this if loading the base VLA checkpoint (not fine-tuned).\n"
                        "Otherwise, you may encounter errors when calling predict_action() due to missing unnorm_key."
                    )
                    
            elif is_embodied_model and self.config.embodied.embodied_type == "openvla":
                # OpenVLA: Use flash_attention_2 for efficiency
                logger.info("Loading OpenVLA model (with flash_attention_2)")
                actor_module = actor_module_class.from_pretrained(
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    config=actor_model_config,
                    trust_remote_code=trust_remote_code,
                )
            else:
                # Default loading for non-VLA models
                actor_module = actor_module_class.from_pretrained(
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    config=actor_model_config,
                    trust_remote_code=trust_remote_code,
                )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)
            
            if not is_embodied_model:
                from siirl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(
                    model=actor_module,
                    use_remove_padding=use_remove_padding,
                    ulysses_sp_size=self.ulysses_sequence_parallel_size,
                    use_fused_kernels=use_fused_kernels,
                )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if self._is_lora:
                logger.info("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {"task_type": TaskType.CAUSAL_LM, 
                               "r": self.config.model.lora_rank, 
                               "lora_alpha": self.config.model.lora_alpha, 
                               "target_modules": convert_to_regular_types(self.config.model.target_modules), "bias": "none"}
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
                
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.param_dtype)
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.reduce_dtype)
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.buffer_dtype)
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.wrap_policy, is_lora=self.config.model.lora_rank > 0)

        if self._is_rollout and self.config.rollout.name == "hf" and not is_embodied_model:
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        if self.rank == 0:
            logger.info(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == Role.Actor else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == Role.Actor and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == Role.Actor else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        # TODO: add more optimizer args into config
        if role == Role.Actor and optim_config is not None:
            from siirl.utils.model_utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.betas,
                weight_decay=optim_config.weight_decay,
            )

            total_steps = optim_config.total_training_steps
            num_warmup_steps = int(optim_config.lr_warmup_steps)
            warmup_style = optim_config.warmup_style
            min_lr_ratio = optim_config.min_lr_ratio if optim_config.min_lr_ratio else 0.0
            num_cycles = optim_config.num_cycles
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.lr_warmup_steps_ratio
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                logger.info(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, 
                                                                     num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, 
                                                                     num_cycles=num_cycles)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _prepare_and_load_model(
        self,
        model_path: str,
        fsdp_config: FSDPArguments,
        override_model_config: DictConfig,
        trust_remote_code: bool,
        role: Role,
    ):
        """
        Prepare configuration and load model from pretrained checkpoint.
        
        Steps:
            1. Register embodied model classes if needed
            2. Determine torch dtype
            3. Load and configure model config
            4. Load model from pretrained
            5. Apply model-specific setup (e.g., OpenVLA-OFT)
        
        Args:
            model_path: Path to model checkpoint
            fsdp_config: FSDP configuration
            override_model_config: Config overrides
            trust_remote_code: Whether to trust remote code
            role: Role (Actor or RefPolicy)
        
        Returns:
            Tuple of (model, model_config, torch_dtype)
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq
        from siirl.utils.model_utils.model import get_generation_config, update_model_config
        from siirl.utils.model_utils.torch_dtypes import PrecisionType
        
        is_embodied = self.config.model.model_type == "embodied"
        
        # Register embodied model classes
        if is_embodied:
            self._register_embodied_model(model_path, trust_remote_code)
        
        # Determine torch dtype
        if fsdp_config.model_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.model_dtype)
        
        # Load model config with appropriate attention implementation
        embodied_type = getattr(self.config.embodied, 'embodied_type', None) if is_embodied else None
        use_flash_attn = not (is_embodied and embodied_type == "openvla-oft")
        
        config_kwargs = {"trust_remote_code": trust_remote_code}
        if use_flash_attn:
            config_kwargs["attn_implementation"] = "flash_attention_2"
        
        model_config = AutoConfig.from_pretrained(model_path, **config_kwargs)
        
        # Initialize flops counter for reference policy
        if self._is_ref:
            self.flops_counter = FlopsCounter(model_config, forward_only=True)
        
        # Apply model-specific patches
        if getattr(model_config, "model_type", None) == "kimi_vl":
            model_config.text_config.topk_method = "greedy"
        
        # Load and update generation config
        self.generation_config = get_generation_config(model_path, trust_remote_code)
        
        override_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_kwargs.update(override_model_config)
        update_model_config(model_config, override_config_kwargs=override_kwargs)
        
        # Load model with appropriate context manager
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings,
            mesh=self.device_mesh
        )
        
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Determine model class
            model_class = self._get_model_class(model_config, is_embodied)
            
            # Load model based on type
            if is_embodied and embodied_type == "openvla-oft":
                logger.info("Loading OpenVLA-OFT model (without flash_attention_2)")
                model = model_class.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    config=model_config,
                    trust_remote_code=trust_remote_code,
                )
                self._setup_openvla_oft_model(model, model_path)
                
            elif is_embodied and embodied_type == "openvla":
                logger.info("Loading OpenVLA model (with flash_attention_2)")
                model = model_class.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    config=model_config,
                    trust_remote_code=trust_remote_code,
                )
            else:
                model = model_class.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    config=model_config,
                    trust_remote_code=trust_remote_code,
                )
        
        return model, model_config, torch_dtype

    def _get_model_class(self, model_config, is_embodied: bool):
        """Determine the appropriate model class based on config."""
        from transformers import AutoModelForCausalLM, AutoModelForVision2Seq
        
        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys() or is_embodied:
            return AutoModelForVision2Seq
        elif type(model_config).__name__ == "configuration_internvl_chat.InternVLChatConfig":
            from siirl.models.transformers.internvl_chat import InternVLChatModel
            logger.info("Using InternVLChatModel for internvl")
            return InternVLChatModel
        else:
            return AutoModelForCausalLM

    def _register_embodied_model(self, model_path: str, trust_remote_code: bool):
        """Register embodied model classes to transformers registry."""
        from transformers import AutoConfig, AutoModelForVision2Seq, AutoImageProcessor, AutoProcessor
        
        embodied_type = self.config.embodied.embodied_type
        
        if embodied_type not in ["openvla-oft", "openvla"]:
            raise ValueError(f"Unsupported embodied type: {embodied_type}")
        
        # Import based on type
        module_name = embodied_type.replace("-", "_")
        config_module = f"siirl.models.embodied.{module_name}.configuration_prismatic"
        model_module = f"siirl.models.embodied.{module_name}.modeling_prismatic"
        processor_module = f"siirl.models.embodied.{module_name}.processing_prismatic"
        
        # Dynamic import
        from importlib import import_module
        config_mod = import_module(config_module)
        model_mod = import_module(model_module)
        processor_mod = import_module(processor_module)
        
        # Register classes
        AutoConfig.register("openvla", config_mod.OpenVLAConfig)
        AutoImageProcessor.register(config_mod.OpenVLAConfig, processor_mod.PrismaticImageProcessor)
        AutoProcessor.register(config_mod.OpenVLAConfig, processor_mod.PrismaticProcessor)
        AutoModelForVision2Seq.register(config_mod.OpenVLAConfig, model_mod.OpenVLAForActionPrediction)
        
        # Update automap on rank 0 (with file locking for safety)
        # Note: update_auto_map now includes retry logic and atomic writes
        if self.rank == 0:
            try:
                from siirl.utils.embodied.openvla_utils import update_auto_map, check_model_logic_mismatch
                logger.info(f"[rank-{self.rank}] Updating auto_map for {model_path}")
                update_auto_map(model_path)
                check_model_logic_mismatch(model_path)
                logger.info(f"[rank-{self.rank}] Successfully updated auto_map")
            except Exception as e:
                logger.error(f"[rank-{self.rank}] Failed to update auto_map: {e}")
                raise
        
        # Synchronize all ranks before proceeding
        torch.distributed.barrier()

    def _setup_openvla_oft_model(self, model: torch.nn.Module, model_path: str):
        """Apply OpenVLA-OFT specific configurations."""
        import json
        
        # Setup multi-camera support
        if hasattr(model, 'vision_backbone'):
            num_images = getattr(self.config.embodied, 'num_images_in_input', 1)
            model.vision_backbone.set_num_images_in_input(num_images)
            logger.info(f"Set vision_backbone.num_images_in_input={num_images}")
        
        # Load dataset statistics for action normalization
        stats_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path, "r") as f:
                model.norm_stats = json.load(f)
            logger.info(f"Loaded dataset_statistics.json with {len(model.norm_stats)} task(s)")
        else:
            logger.warning(
                "No dataset_statistics.json found for OpenVLA-OFT. "
                "This is expected for base checkpoints but may cause errors for fine-tuned models."
            )

    def _apply_model_modifications(
        self,
        model: torch.nn.Module,
        use_liger: bool,
        use_remove_padding: bool,
        use_fused_kernels: bool,
        enable_gradient_checkpointing: bool,
        torch_dtype: torch.dtype,
    ):
        """
        Apply various model modifications and optimizations.
        
        Modifications include:
            1. Liger kernel optimization
            2. Monkey patch (for remove padding and Ulysses SP)
            3. Ensure correct dtype for all parameters
            4. Gradient checkpointing
            5. LoRA adapter
        
        Args:
            model: Model to modify
            use_liger: Whether to apply Liger kernel
            use_remove_padding: Whether to use remove padding
            use_fused_kernels: Whether to use fused kernels
            enable_gradient_checkpointing: Whether to enable gradient checkpointing
            torch_dtype: Target torch dtype
        
        Returns:
            Modified model
        """
        is_embodied = self.config.model.model_type == "embodied"
        
        # Apply Liger kernel optimization
        if use_liger:
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
            _apply_liger_kernel_to_instance(model=model)
            logger.info("Applied Liger kernel to model")
        
        # Apply monkey patch for non-embodied models
        if not is_embodied:
            from siirl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(
                model=model,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
            )
        
        # Ensure dtype consistency
        model.to(torch_dtype)
        
        # Enable gradient checkpointing
        if enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Enabled gradient checkpointing")
        
        # Apply LoRA adapter
        if self._is_lora:
            logger.info("Applying LoRA to model")
            model.enable_input_require_grads()
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "target_modules": convert_to_regular_types(self.config.model.target_modules),
                "bias": "none"
            }
            model = get_peft_model(model, LoraConfig(**lora_config))
        
        return model

    def _setup_fsdp_wrapper(
        self,
        model: torch.nn.Module,
        fsdp_config: FSDPArguments,
        role: Role,
        enable_activation_offload: bool,
        enable_gradient_checkpointing: bool,
    ):
        """
        Wrap model with FSDP (Fully Sharded Data Parallel).
        
        Steps:
            1. Configure mixed precision
            2. Get wrap policy
            3. Wrap model based on strategy (fsdp/fsdp2)
            4. Apply activation offload if needed
        
        Args:
            model: Model to wrap
            fsdp_config: FSDP configuration
            role: Role (Actor or RefPolicy)
            enable_activation_offload: Whether to enable activation offload
            enable_gradient_checkpointing: Whether gradient checkpointing is enabled
        
        Returns:
            FSDP wrapped model
        """
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from siirl.utils.model_utils.torch_dtypes import PrecisionType
        
        is_embodied = self.config.model.model_type == "embodied"
        
        # Configure mixed precision
        mixed_precision_config = fsdp_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.param_dtype)
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.reduce_dtype)
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.buffer_dtype)
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
        
        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype
        )
        
        # Get wrap policy
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=model,
            config=fsdp_config.wrap_policy,
            is_lora=self.config.model.lora_rank > 0
        )
        
        # Special case: HFRollout with Gemma
        if self._is_rollout and self.config.rollout.name == "hf" and not is_embodied:
            auto_wrap_policy = None
        
        if self.rank == 0:
            logger.info(f"wrap_policy: {auto_wrap_policy}")
        
        # Prepare FSDP mesh and strategy
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        fsdp_strategy = self.config.actor.strategy
        
        # Wrap model based on FSDP strategy
        if fsdp_strategy == "fsdp":
            # FSDP v1
            cpu_offload = None if role == Role.Actor else CPUOffload(offload_params=True)
            model_fsdp = FSDP(
                model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            # FSDP v2
            assert CPUOffloadPolicy is not None, \
                "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True
            )
            
            if role == Role.Actor and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == Role.Actor else CPUOffloadPolicy(pin_memory=True)
            
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = model.state_dict()
            apply_fsdp2(model, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(model, full_state, fsdp_mesh, cpu_offload)
            model_fsdp = model
        else:
            raise NotImplementedError(f"FSDP strategy '{fsdp_strategy}' not implemented")
        
        # Apply activation offload
        if enable_activation_offload:
            enable_activation_offloading(
                model_fsdp, fsdp_strategy, enable_gradient_checkpointing
            )
        
        return model_fsdp

    def _create_optimizer_and_scheduler(
        self,
        model: torch.nn.Module,
        optim_config: Optional[OptimizerArguments],
        role: Role,
    ):
        """
        Create optimizer and learning rate scheduler.
        
        Only creates when role is Actor and optim_config is provided.
        
        Args:
            model: Model to create optimizer for
            optim_config: Optimizer configuration
            role: Role (Actor or RefPolicy)
        
        Returns:
            Tuple of (optimizer, lr_scheduler), returns (None, None) if not needed
        """
        if role != Role.Actor or optim_config is None:
            return None, None
        
        from torch import optim
        from siirl.utils.model_utils.torch_functional import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup
        )
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optim_config.lr,
            betas=optim_config.betas,
            weight_decay=optim_config.weight_decay,
        )
        
        # Calculate warmup steps
        total_steps = optim_config.total_training_steps
        num_warmup_steps = int(optim_config.lr_warmup_steps)
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = optim_config.lr_warmup_steps_ratio
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
        
        if self.rank == 0:
            logger.info(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")
        
        # Create learning rate scheduler
        warmup_style = optim_config.warmup_style
        if warmup_style == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps
            )
        elif warmup_style == "cosine":
            min_lr_ratio = optim_config.min_lr_ratio if optim_config.min_lr_ratio else 0.0
            num_cycles = optim_config.num_cycles
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles
            )
        else:
            raise NotImplementedError(f"Warmup style '{warmup_style}' is not supported")
        
        return optimizer, lr_scheduler

    def _build_rollout(self, trust_remote_code=False):
        from siirl.utils.model_utils.model import get_generation_config

        local_path = copy_to_local(self.config.model.path)
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        # TODO(sgm): support FSDP hybrid shard for larger model
        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            if self.config.model.model_type == "embodied":
                from siirl.engine.rollout.embodied_rollout import EmbodiedHFRollout
                rollout = EmbodiedHFRollout(module=None, config=self.config)
            else: 
                from siirl.engine.rollout import HFRollout
                rollout = HFRollout(module=None, config=self.config)

        elif rollout_name == "vllm":
            from siirl.engine.rollout.vllm_rollout import vllm_mode, vLLMRollout

            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.use_shm)
            lora_kwargs = {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}
            # lora_kwargs = {}
            if vllm_mode == "customized":
                rollout = vLLMRollout(actor_module=self.actor_module_fsdp, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, trust_remote_code=trust_remote_code, **lora_kwargs)
            elif vllm_mode == "spmd":
                from siirl.engine.rollout.vllm_rollout import vLLMAsyncRollout

                vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
                rollout = vllm_rollout_cls(model_path=local_path, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, trust_remote_code=trust_remote_code, **lora_kwargs)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            if self.device_mesh.mesh.numel() == 1:
                self.config.rollout.load_format = "dummy_hf"

        elif rollout_name == "sglang":
            from siirl.engine.rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to siirl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            # from siirl.engine.sharding_manager.fsdp_sglang import MultiAgentFSDPSGLangShardingManager

            local_path = copy_to_local(self.config.model.path)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                trust_remote_code=trust_remote_code,
            )

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, None

    def init_model(self):
        from siirl.engine.actor import DataParallelPPOActor, RobDataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.external_lib)

        override_model_config = self.config.model.override_config
        use_remove_padding = self.config.model.use_remove_padding
        use_fused_kernels = self.config.model.use_fused_kernels
        use_shm = self.config.model.use_shm

        tokenizer_module = load_tokenizer(model_args=self.config.model)
        self.tokenizer, self.processor = tokenizer_module["tokenizer"], tokenizer_module["processor"]

        if self._is_actor:
            optim_config = self.config.actor.optim
            fsdp_config = self.config.actor.fsdp_config

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.enable_gradient_checkpointing,
                trust_remote_code=self.config.model.trust_remote_code,
                use_liger=self.config.model.use_liger,
                role=Role.Actor,
                enable_activation_offload=self.config.model.enable_activation_offload,
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        # load from checkpoint
        if self._is_actor:
            self.config.actor.use_remove_padding = use_remove_padding
            self.config.actor.use_fused_kernels = use_fused_kernels
            
            # Select appropriate Actor class based on model type and pass embodied parameters
            is_embodied_model = self.config.model.model_type == "embodied"
            if is_embodied_model:
                self.config.actor.embodied_type = self.config.embodied.embodied_type
                self.config.actor.action_token_len = self.config.embodied.action_token_len
                self.config.actor.action_chunks_len = self.config.embodied.action_chunks_len
                
                from siirl.workers.actor.embodied_actor import RobDataParallelPPOActor
                ActorClass = RobDataParallelPPOActor
            else:
                ActorClass = DataParallelPPOActor
            
            self.actor = ActorClass(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            from transformers import AutoConfig

            local_path = copy_to_local(self.config.model.path)
            self.actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=self.config.model.trust_remote_code)
            self.flops_counter = FlopsCounter(self.actor_model_config, forward_only=True)
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.trust_remote_code)

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.trust_remote_code,
                use_liger=self.config.model.use_liger,
                role=Role.RefPolicy,
            )[0]
            self.config.ref.use_remove_padding = use_remove_padding
            self.config.ref.use_fused_kernels = use_fused_kernels
            
            # Pass embodied parameters to RefPolicy for embodied models and select class
            is_embodied_model = self.config.model.model_type == "embodied"
            if is_embodied_model:
                self.config.ref.embodied_type = self.config.embodied.embodied_type
                self.config.ref.action_token_len = self.config.embodied.action_token_len
                self.config.ref.action_chunks_len = self.config.embodied.action_chunks_len
                
                from siirl.workers.actor.embodied_actor import RobDataParallelPPOActor
                RefPolicyClass = RobDataParallelPPOActor
            else:
                RefPolicyClass = DataParallelPPOActor
            
            self.ref_policy = RefPolicyClass(
                config=self.config.ref,
                actor_module=self.ref_module_fsdp
            )

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp, optimizer=self.actor.actor_optimizer, lr_scheduler=self.actor_lr_scheduler, processing_class=self.processor if self.processor is not None else self.tokenizer, checkpoint_contents=self.config.actor.checkpoint.contents, tokenizer=self.tokenizer
            )

    def update_actor(self, data: TensorDict):
        # Support all hardwares
        data = data.to(get_device_id())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops
            metrics["perf/delta_time/actor"] = delta_time
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            data["metrics"] = NonTensorData(metrics)
            processed_data = self.ulysses_sharding_manager.postprocess_data(data=data)
            processed_data = processed_data.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return processed_data

    def generate_sequences(self, prompts: TensorDict):
        # Support all hardwares
        prompts = prompts.to(get_device_id())
        assert self._is_rollout
        prompts["eos_token_id"] = NonTensorData(self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id)
        prompts["pad_token_id"] = NonTensorData(self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id)
        
        with self.rollout_sharding_manager:
            if self.config.rollout.name == "sglang_async":
                from siirl.engine.rollout.sglang_rollout import AsyncSGLangRollout

                if isinstance(self.rollout, AsyncSGLangRollout) and hasattr(self.rollout, "_tool_schemas") and len(self.rollout._tool_schemas) > 0:
                    output = self.rollout.generate_sequences_with_tools(prompts=prompts)
                else:
                    output = self.rollout.generate_sequences(prompts=prompts)
            else:
                with Timer(name="generate_sequences", logger=None) as timer:
                    output = self.rollout.generate_sequences(prompts=prompts)
                total_input_tokens = output["total_input_tokens"] if "total_input_tokens" in output else 0
                total_output_tokens = output["total_output_tokens"] if "total_output_tokens" in output else 0
                delta_time = timer.last

                # Calculate correct batch_seqlens for MFU computation
                # Get batch size from prompts
                batch_size = prompts["input_ids"].shape[0] if "input_ids" in prompts else 1
                # Calculate average sequence length per sample (prompt + response)
                avg_seq_len = (total_input_tokens + total_output_tokens) / batch_size if batch_size > 0 else 0
                # Create batch_seqlens list with each sample's average length
                batch_seqlens = [int(avg_seq_len)] * batch_size

                estimated_flops, promised_flops = self.flops_counter.estimate_flops(batch_seqlens, delta_time)
                metrics = {}
                # MFU should not be divided by TP size - it's already per-GPU
                metrics["perf/mfu/rollout"] = estimated_flops / promised_flops
                metrics["perf/delta_time/rollout"] = delta_time
                output["metrics"] = NonTensorData(metrics, batch_size=None)


        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    def compute_log_prob(self, data: TensorDict):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext
        is_lora = data.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        
        data = data.to(get_device_id())
        
        # we should always recompute old_log_probs when it is HybridEngine
        data["micro_batch_size"] = NonTensorData(self.config.rollout.log_prob_micro_batch_size_per_gpu)  
        data["max_token_len"] = NonTensorData(self.config.rollout.log_prob_max_token_len_per_gpu)
        data["use_dynamic_bsz"] = NonTensorData(self.config.rollout.log_prob_use_dynamic_bsz)
        data["temperature"] = NonTensorData(self.config.rollout.temperature)
        data["pad_token_id"] = NonTensorData(self.tokenizer.pad_token_id)

        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with Timer(name="compute_actor_log_prob", logger=None) as timer, adapter_ctx:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            delta_time = timer.last
            global_num_tokens = data["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics = {
                # actor forward
                "perf/mfu/actor_log_prob": estimated_flops / promised_flops / 3,
                "perf/delta_time/actor_log_prob": delta_time,
            }
            data["old_log_probs"] = output
            if entropys is not None:
                data["entropys"] = entropys 
            data["metrics"] = NonTensorData(metrics)
            processed_data = self.ulysses_sharding_manager.postprocess_data(data)

        processed_data = processed_data.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.group_world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        return processed_data

    def compute_ref_log_prob(self, data: TensorDict):    
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data["is_lora"] = NonTensorData(True)
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = TensorDict({"ref_log_prob": data["old_log_probs"]})
            return data
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model
        # Support all hardwares
        data = data.to(get_device_id())

        metrics = {}
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        data["temperature"] = NonTensorData(self.config.rollout.temperature)
        data["max_token_len"] = NonTensorData(self.config.ref.log_prob_max_token_len_per_gpu)
        data["use_dynamic_bsz"] = NonTensorData(self.config.ref.log_prob_use_dynamic_bsz)
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with Timer(name="compute_log_prob", logger=None) as timer:
                output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            delta_time = timer.last
            global_num_tokens = data["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/ref"] = estimated_flops / promised_flops
            metrics["perf/delta_time/ref"] = delta_time
            data["ref_log_prob"] = output
            data["metrics"] = NonTensorData(metrics)
            processed_data = self.ulysses_sharding_manager.postprocess_data(data)

        processed_data = processed_data.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.group_world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return processed_data

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if torch.distributed.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if torch.distributed.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                if torch.distributed.get_rank() == 0:
                    logger.info(f"[rank-{self.rank}]: Save LoRA Adapter Error ({e})")

            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                logger.info(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    def set_rollout_sharding_manager(self, sharding_manager):
        self.rollout_sharding_manager = sharding_manager
        self.rollout.sharding_manager = sharding_manager


class CriticWorker(Worker):
    def __init__(self, config: CriticArguments, process_group: ProcessGroup):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_nccl_backend())
        self.config = config
        world_size = torch.distributed.get_world_size(group=process_group)
        self.group_world_size = world_size
        # build device mesh for Ulysses Sequence Parallel
        self.device_mesh = create_device_mesh_from_group(process_group=process_group, fsdp_size=self.config.model.fsdp_config.fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = create_device_mesh_from_group(process_group=process_group, sp_size=self.ulysses_sequence_parallel_size)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= world_size // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= world_size // self.ulysses_sequence_parallel_size
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
        self._is_lora = self.config.model.lora_rank > 0

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        from siirl.utils.model_utils.model import print_model_size
        from siirl.utils.model_utils.torch_dtypes import PrecisionType

        use_shm = config.model.use_shm
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_module = load_tokenizer(model_args=self.config.model)
        self.tokenizer, self.processor = tokenizer_module["tokenizer"], tokenizer_module["processor"]

        override_config = self.config.model.override_config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        # if self.rank == 0:
        #     logger.info(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.model_dtype
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification

        critic_model_config = AutoConfig.from_pretrained(local_path, attn_implementation="flash_attention_2", trust_remote_code=config.model.trust_remote_code)
        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=critic_model_config,
                trust_remote_code=config.model.trust_remote_code,
            )

            use_remove_padding = config.model.use_remove_padding

            # Apply monkey patch for performance optimizations
            from siirl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.enable_gradient_checkpointing:
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            logger.info("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "target_modules": convert_to_regular_types(self.config.model.target_modules),
                "bias": "none",
            }
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.param_dtype)
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.reduce_dtype)
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.buffer_dtype)
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy, is_lora=self.config.model.lora_rank > 0)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=False,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.enable_activation_offload:
            enable_gradient_checkpointing = config.model.enable_gradient_checkpointing
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.betas,
            weight_decay=config.optim.weight_decay,
        )

        total_steps = config.optim.total_training_steps
        num_warmup_steps = int(config.optim.lr_warmup_steps)
        warmup_style = config.optim.warmup_style
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.lr_warmup_steps_ratio
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            logger.info(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from siirl.utils.model_utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if warmup_style == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps)
        elif warmup_style == "cosine":
            critic_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.external_lib)

        from siirl.engine.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module, optimizer=self.critic_optimizer, lr_scheduler=self.critic_lr_scheduler, processing_class=self.processor if self.processor is not None else self.tokenizer, checkpoint_contents=self.config.checkpoint.contents, tokenizer=self.tokenizer
        )

    def compute_values(self, data: TensorDict):
        # Support all hardwares
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data["micro_batch_size"] = NonTensorData(micro_batch_size)
        data["max_token_len"] = NonTensorData(self.config.forward_max_token_len_per_gpu)
        data["use_dynamic_bsz"] = NonTensorData(self.config.use_dynamic_bsz)
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            data["values"] = values
            processed_data = self.ulysses_sharding_manager.postprocess_data(data=data)

        processed_data = processed_data.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return processed_data

    def update_critic(self, data: TensorDict):
        # Support all hardwares
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            data["metrics"] = NonTensorData(metrics)
            processed_data = self.ulysses_sharding_manager.postprocess_data(data=data)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        processed_data = processed_data.to("cpu")
        return processed_data

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config: RewardModelArguments, process_group: ProcessGroup):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_nccl_backend())
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size(group=process_group)

        self.device_mesh = create_device_mesh_from_group(process_group=process_group, fsdp_size=self.config.model.fsdp_config.fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = create_device_mesh_from_group(process_group=process_group, sp_size=self.ulysses_sequence_parallel_size)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.use_remove_padding

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= world_size
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config: RewardModelArguments):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.use_shm
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer)
            input_tokenizer_module = load_tokenizer(path=input_tokenizer_local_path)
            self.input_tokenizer = input_tokenizer_module["tokenizer"]
            tokenizer_module = load_tokenizer(model_args=self.config.model)
            self.tokenizer, self.processor = tokenizer_module["tokenizer"], tokenizer_module["processor"]

        trust_remote_code = config.model.trust_remote_code
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )            
            # Apply monkey patch for performance optimizations
            from siirl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config.wrap_policy)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=False,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.external_lib)
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        if is_cuda_available:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        elif is_npu_available:
            from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

        from siirl.utils.model_utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False)  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: TensorDict, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: TensorDict):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                logger.info(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.max_length
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.truncation,
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return TensorDict.from_dict(rm_inputs)

    def compute_rm_score(self, data: TensorDict):
        import itertools

        from siirl.utils.model_utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = TensorDict.from_dict(rm_inputs)

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            data.batch["rm_scores"] = token_level_scores
            processed_data = self.ulysses_sharding_manager.postprocess_data(data=data)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        processed_data = processed_data.to("cpu")
        return processed_data


# ================================= Async related workers =================================
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

    def generate_sequences(self, prompts: TensorDict):
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

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
        
            
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

