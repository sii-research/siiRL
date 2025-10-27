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
from typing import Any, Dict


@dataclass
class ProfilerArguments:
    enable: bool = field(default=False, metadata={"help": "Whether to enable profiling"})
    save_path: str = field(default="./prof_data", metadata={"help": "Storage path for collected data"})
    level: str = field(default="level1", metadata={"help": "Collection level-options are level_none, level0, level1, and level2"})
    with_memory: bool = field(default=False, metadata={"help": "Whether to enable memory analysis"})
    record_shapes: bool = field(default=False, metadata={"help": "Whether to record tensor shapes"})
    with_npu: bool = field(default=True, metadata={"help": "Whether to collect device-side performance data"})
    with_cpu: bool = field(default=False, metadata={"help": "Whether to collect host-side performance data"})
    with_module: bool = field(default=False, metadata={"help": "Whether to record framework-layer Python call stack information"})
    with_stack: bool = field(default=False, metadata={"help": "Whether to record operator call stack information"})
    analysis: bool = field(default=True, metadata={"help": "Enables automatic data parsing"})
    discrete: bool = field(default=False, metadata={
        "help": "True for each task has its own database, False for all tasks in one training step share one database"})
    roles: list[str] = field(default_factory=lambda: ["generate","compute_reward"], metadata={
        "help": "Used for discrete mode, optional values: generate, compute_reward, compute_old_log_prob, compute_ref_log_porb, compute_value, compute_advantage, train_critic, train_actor"})
    all_ranks: bool = field(default=False, metadata={"help": "Whether to profile all ranks"})
    ranks: list[int] = field(default_factory=lambda: [0], metadata={"help": "The ranks that will be profiled. [] or [0,1,...]"})
    profile_steps: list[int] = field(default_factory=lambda: [0], metadata={"help": "The steps that will be profiled"})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
