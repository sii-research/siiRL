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
from typing import Optional, Dict, List, Any
from siirl.global_config.params.data_args import DataArguments
from siirl.global_config.params.model_args import (
    ActorRolloutRefArguments,
    CriticArguments,
    RewardModelArguments,
    AlgorithmArguments,
)
from siirl.global_config.params.dag_args import DagArguments
from siirl.global_config.params.profiler_args import ProfilerArguments


@dataclass
class TrainingArguments:
    total_epochs: int = field(default=30, metadata={"help": "Total training epochs"})
    total_training_steps: Optional[int] = field(default=None, metadata={"help": "Override training steps"})
    project_name: str = field(default="siirl_examples", metadata={"help": "Project name"})
    experiment_name: str = field(default="gsm8k", metadata={"help": "Experiment name"})
    logger: List[str] = field(
        default_factory=lambda: ["console", "wandb"],
        metadata={"help": "Logging backends"},
    )
    log_val_generations: int = field(default=0, metadata={"help": "Validation samples to log"})
    nnodes: int = field(default=1, metadata={"help": "Number of nodes"})
    n_gpus_per_node: int = field(default=8, metadata={"help": "GPUs per node"})
    save_freq: int = field(default=-1, metadata={"help": "Checkpoint frequency"})
    resume_mode: str = field(default="auto", metadata={"help": "Resume training mode"})
    resume_from_path: bool = field(default=False, metadata={"help": "Resume from specific path"})
    test_freq: int = field(default=-1, metadata={"help": "Testing frequency"})
    critic_warmup: int = field(default=0, metadata={"help": "Critic warmup steps"})
    default_local_dir: str = field(
        default="checkpoints/siirl_examples/gsm8k",
        metadata={"help": "Checkpoint directory"},
    )
    seed: int = field(default=1, metadata={"help": "Train seed param"})
    should_log: bool = field(default=False, metadata={"help": "Should print debug log for training"})
    should_save: bool = field(
        default=False,
        metadata={"help": "Should save tokenized dataset to local disk and exit"},
    )
    val_before_train: bool = field(default=True, metadata={"help": "Whether or not to validate before train"})
    default_hdfs_dir: str = field(default=None, metadata={"help": "Default hdfs dir path for checkpoints"})
    del_local_ckpt_after_load: bool = field(
        default=False,
        metadata={"help": "Whether or not to delete local checkpoints after load"},
    )
    val_only: bool = field(default=False, metadata={"help": "Whether or not just eval only"})
    balance_batch: bool = field(
        default=False,
        metadata={"help": "Whether or not to balance the number of valid tokens on each dp rank."},
    )
    remove_previous_ckpt_in_save: bool = field(
        default=False,
        metadata={"help": "Whether or not to remove previous ckpt in save path."},
    )
    max_actor_ckpt_to_keep: int = field(default=100, metadata={"help": "Maximum number of actor ckpts."})
    max_critic_ckpt_to_keep: int = field(default=100, metadata={"help": "Maximum number of critic ckpts."})
    ray_wait_register_center_timeout: int = field(default=300, metadata={"help": "The timeout for ray worker group to wait for the register center to be ready"})
    validation_data_dir: Optional[str] = field(default=None, metadata={"help": "Validation data directory."})
    rollout_data_dir: Optional[str] = field(default=None, metadata={"help": "Rollout data directory."})
    device: Optional[str] = field(default="cuda", metadata={"help": "Training device."})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CustomRewardArguments:
    path: str = field(default=None, metadata={"help": "Custom reward function import file path"})
    name: str = field(default="compute_score", metadata={"help": "Custom reward function name"})
    reward_kwargs: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class SiiRLArguments:
    data: DataArguments = field(default_factory=DataArguments)
    actor_rollout_ref: ActorRolloutRefArguments = field(default_factory=ActorRolloutRefArguments)
    critic: CriticArguments = field(default_factory=CriticArguments)
    reward_model: RewardModelArguments = field(default_factory=RewardModelArguments)
    algorithm: AlgorithmArguments = field(default_factory=AlgorithmArguments)
    trainer: TrainingArguments = field(default_factory=TrainingArguments)
    custom_reward_function: CustomRewardArguments = field(default_factory=CustomRewardArguments)
    dag: DagArguments = field(default_factory=DagArguments)
    profiler: ProfilerArguments = field(default_factory=ProfilerArguments)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
