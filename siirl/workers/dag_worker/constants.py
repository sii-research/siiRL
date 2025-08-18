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

from typing import Dict, List
from siirl.workers.dag.node import NodeRole


class DAGInitializationError(Exception):
    """Custom exception for failures during DAGWorker initialization."""

    pass


class DAGConstants:
    """Centralized constants to improve maintainability and avoid magic strings."""

    # Worker role mapping
    WORKER_ROLE_MAPPING: Dict[NodeRole, str] = {
        NodeRole.ACTOR: "actor",
        NodeRole.ROLLOUT: "rollout",
        NodeRole.REFERENCE: "ref",
    }
    # Configuration keys
    INTERN_CONFIG: str = "intern_config"
    # Framework strategy names
    FSDP_STRATEGIES: List[str] = ["fsdp", "fsdp2"]
    MEGATRON_STRATEGY: str = "megatron"
    # Metric group order
    METRIC_GROUP_ORDER = ["step", "training", "actor", "critic", "perf", "response_length", "response", "prompt_length", "prompt", "global_seqlen", "timing_s", "timing_per_token_ms", "perf/total_num_tokens", "perf/time_per_step", "perf/throughput"]
