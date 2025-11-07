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

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from siirl.utils.reward_score import embodied


@dataclass
class EnvironmentArgs:
    """Unified configuration for Embodied AI environments."""
    env_type: str = field(
        default="libero",
        metadata={"help": "Environment type: 'libero' or 'maniskill'"}
    )
    env_name: str = field(
        default="libero_10",
        metadata={"help": "Name of the specific environment or task suite to load (e.g., 'libero_spatial', 'PickCube-v1')"}
    )
    num_envs: int = field(
        default=16,
        metadata={"help": "Number of parallel environments to run"}
    )
    max_steps: int = field(
        default=512,
        metadata={"help": "Maximum number of steps per episode"}
    )
    num_trials_per_task: int = field(
        default=50,
        metadata={"help": "Number of trials per task for dataset preparation"}
    )
    num_steps_wait: int = field(
        default=10,
        metadata={"help": "Number of steps to wait before polling for environment completion"}
    )
    model_family: str = field(
        default="openvla",
        metadata={"help": "Model family for environment interaction (e.g., 'openvla')"}
    )


@dataclass
class EmbodiedArguments:
    """Embodied AI-specific configuration for Vision-Language-Action models."""

    # Unified Environment Configuration
    env: EnvironmentArgs = field(
        default_factory=EnvironmentArgs,
        metadata={"help": "Unified environment configuration for Embodied AI tasks"}
    )

    embodied_type: str = field(
        default="openvla", 
        metadata={"help": "Embodied model type: 'openvla' or 'openvla-oft'"}
    )
    model_path: str = field(
        default="openvla/openvla-7b", metadata={"help": "Path to Embodied AI model"}
    )
    video_embedding_model_path: str = field(
        default="~/models/vjepa/vitg-384.pt",
        metadata={"help": "Path to V-JEPA embedding model"},
    )

    action_chunks_len: int = field(
        default=8, metadata={"help": "Number of action chunks per step"}
    )
    action_token_len: int = field(
        default=7, metadata={"help": "Number of action tokens"}
    )

    # Image processing
    embedding_img_size: int = field(
        default=384, metadata={"help": "Image size for video embedding"}
    )
    embedding_enable_fp16: bool = field(
        default=True, metadata={"help": "Enable FP16 for video embedding"}
    )
    # Embedding model configuration
    embedding_model_class: Optional[str] = field(
        default=None, metadata={"help": "Custom embedding model class path"}
    )
    embedding_model_offload: bool = field(
        default=False,
        metadata={"help": "Offload embedding model to CPU when not in use"},
    )
    num_images_in_input: int = field(
        default=1, metadata={"help": "Number of camera views (1=main, 2=main+wrist)"}
    )
    center_crop: bool = field(
        default=True, metadata={"help": "Apply center crop to images"}
    )
    # Action normalization stats (will be populated from config)
    unnorm_key: str = field(
        default="libero_10",
        metadata={"help": "Key for action normalization stats used in un-normalization"},
    )
    # Generation parameters
    temperature: float = field(
        default=1.6, metadata={"help": "Sampling temperature for action generation"}
    )

    n_gpus_per_node: int = field(
        default=8,
        metadata={"help": "Number of GPUs per node"}
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EmbodiedSamplingConfig:
    """Configuration for embodied AI dynamic sampling and filtering (similar to DAPO filter_groups)."""
    
    filter_accuracy: bool = field(
        default=False,
        metadata={"help": "Enable accuracy-based filtering for embodied tasks (verl: True)"}
    )
    accuracy_lower_bound: float = field(
        default=0.0,
        metadata={"help": "Minimum success rate threshold for keeping prompts (verl: 0.1)"}
    )
    accuracy_upper_bound: float = field(
        default=1.0,
        metadata={"help": "Maximum success rate threshold for keeping prompts (verl: 0.9)"}
    )
    filter_truncated: bool = field(
        default=False,
        metadata={"help": "Filter out truncated episodes (uses env.max_steps for truncation detection)"}
    )
    oversample_factor: int = field(
        default=1,
        metadata={"help": "Oversample factor for data filtering (verl: 1)"}
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "EnvironmentArgs",
    "EmbodiedArguments",
    "EmbodiedSamplingConfig"
]
