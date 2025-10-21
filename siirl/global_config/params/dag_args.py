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
    workflow_path: Optional[str] = field(
        default=None,
        metadata={"help": "Workflow DAG config file (YAML, legacy mode). Consider using custom_pipeline_fn instead."}
    )
    custom_pipeline_fn: Optional[str] = field(
        default=None,
        metadata={
            "help": "Custom pipeline function path in format 'module.path:function_name'. "
                    "Example: 'examples.custom_pipeline_example.custom_grpo:grpo_with_custom_reward'. "
                    "If not specified, built-in pipelines will be used based on algorithm type."
        }
    )
    env_enable: bool = field(default=False, metadata={"help": "Enable environment"})
    environment_path: Optional[str] = field(default=None, metadata={"help": "Environment config file"})
    enable_perf: bool = field(default=False, metadata={"help": "Enable all ranks performance profiling table"})
    backend_threshold: int = field(default=256, metadata={"help": "World size threshold for backend selection"})
