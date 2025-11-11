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
ShardingManager for FSDP + HF Rollout (including EmbodiedHFRollout).
Manages model loading/offloading between training (actor) and inference (rollout).
"""

from loguru import logger
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from siirl.utils.extras.device import get_torch_device
from siirl.utils.model_utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from siirl.workers.sharding_manager.base import BaseShardingManager


class FSDPHFShardingManager(BaseShardingManager):
    """
    ShardingManager for FSDP + HuggingFace Rollout.
    
    This manager handles model offloading for HF-based rollout (including EmbodiedHFRollout).
    - In __enter__: Load actor model (and embedding model if needed) to GPU before rollout
    - In __exit__: Offload actor model (and embedding model) to CPU after rollout
    
    This follows the same pattern as MultiAgentFSDPVLLMShardingManager and 
    MultiAgentFSDPSGLangShardingManager for consistency.
    """
    
    def __init__(
        self, 
        module: FSDP, 
        rollout, 
        offload_param: bool = False,
        offload_embedding: bool = False
    ):
        """
        Initialize FSDP HF Sharding Manager.
        
        Args:
            module: The FSDP-wrapped actor model (actor_module_fsdp)
            rollout: The rollout object (HFRollout or EmbodiedHFRollout)
            offload_param: Whether to offload actor model parameters
            offload_embedding: Whether to offload embedding model (for EmbodiedHFRollout)
        """
        self.module = module
        self.rollout = rollout
        self.offload_param = offload_param
        self.offload_embedding = offload_embedding
        
        # Track state
        self.is_asleep = False  # Model starts on GPU after initialization
        
        logger.info(
            f"FSDPHFShardingManager initialized: "
            f"offload_param={offload_param}, offload_embedding={offload_embedding}"
        )
    
    def __enter__(self):
        """
        Called before rollout generation.
        Load models to GPU if they were offloaded.
        """
        if not self.is_asleep:
            # Models already on GPU (first time or previous rollout didn't offload)
            return
        
        # 1. Load actor model to GPU
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        
        # 2. Load embedding model to GPU (for EmbodiedHFRollout)
        if self.offload_embedding:
            self.rollout.embedding_model.load_to_device()
        
        self.is_asleep = False
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Called after rollout generation.
        Offload models to CPU to free GPU memory.
        """
        if self.is_asleep:
            # Already offloaded
            return
        
        # 1. Offload embedding model first (for EmbodiedHFRollout)
        if self.offload_embedding:
            self.rollout.embedding_model.offload_to_host()
        
        # 2. Offload actor model to CPU
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        
        # 3. Clear cache
        get_torch_device().empty_cache()
        
        self.is_asleep = True

