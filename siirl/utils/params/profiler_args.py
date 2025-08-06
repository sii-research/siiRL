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
    save_path: str = field(default="./prof_data", meta_data={"help", "Storage path for collect data"})
    level: str = field(default="level0", meta_data={"help", ""})
    with_memory: bool = field(default=False, meta_data={"help", "Whether to enable memory analysis"})
    record_shapes: bool = field(default=False, meta_data={"help", "Whether to record tensor shapes"})
    with_npu: bool = field(default=False, meta_data={"help", "Whether to collect device-side performance data"})
    with_cpu: bool = field(default=False, meta_data={"help", "Whether to collect host-side performace data"})
    with_module: bool = field(default=Flase, meta_data={"help", "Whether to record framework-layer Python call stack information"})
    with_stack: bool = field(default=False, meta_data={"help", "Whether to record operator call stack infomation"})
    analysis: bool = field(default=False, meta_data={"help", "Enables automatic data parsing"})
    discrete: bool = field(default=False, meta_data={"help", "TODO"})
    roles: str = field(default="all", meta_data={"help", "TODO"})
    all_ranks: bool = field(default=False, meta_data={"help", "TODO"})
    ranks: list[int] = field(default_factory=lambda: [0], meta_data={"help", "TODO"})
    profile_step: list[int] = field(default_factory=lambda: [0], meta_data={"help", "TODO"})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)