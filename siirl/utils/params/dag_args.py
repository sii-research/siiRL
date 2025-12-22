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
from typing import Optional


@dataclass
class DagArguments:
    workflow_path: Optional[str] = field(default=None, metadata={"help": "Workerflow Dag config file"})
    env_enable: bool = field(default=False, metadata={"help": "Enable environment"})
    environment_path: Optional[str] = field(default=None, metadata={"help": "Environment config file"})
    enable_perf: bool = field(default=False, metadata={"help": "Enable all ranks performance profiling table"})
    backend_threshold: int = field(default=256, metadata={"help": "World size threshold for backend selection"})
