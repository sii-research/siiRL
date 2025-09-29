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

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import ray
from loguru import logger
from torch.distributed import ProcessGroup

from siirl.models.loader import TokenizerModule
from siirl.execution.scheduler.process_group_manager import ProcessGroupManager
from siirl.global_config.params import SiiRLArguments
from siirl.engine.base_worker import Worker
from siirl.execution.dag import TaskGraph
from siirl.execution.dag.node import NodeRole
from siirl.data_coordinator import DataProto

from .constants import DAGInitializationError
from .mixins.execution_mixin import ExecutionMixin
from .mixins.initialization_mixin import InitializationMixin
from .mixins.node_executors_mixin import NodeExecutorsMixin
from .mixins.utilities_mixin import UtilitiesMixin
from .mixins.validation_mixin import ValidationMixin


class DAGWorker(InitializationMixin, ExecutionMixin, NodeExecutorsMixin, ValidationMixin, UtilitiesMixin, Worker):
    """
    Orchestrates a Directed Acyclic Graph (DAG) of tasks for distributed training,
    managing the setup, initialization, and workflow for a specific rank.
    """

    def __init__(
        self,
        config: SiiRLArguments,
        process_group_manager: ProcessGroupManager,
        taskgraph_mapping: Dict[int, TaskGraph],
        data_buffers: List["ray.actor.ActorHandle"]
    ):
        super().__init__()
        self.config = config
        self.process_group_manager = process_group_manager
        self.taskgraph_mapping = taskgraph_mapping
        self.data_buffers = data_buffers
        self.enable_perf = os.environ.get("SIIRL_ENABLE_PERF", "0") == "1" or config.dag.enable_perf

        # State attributes
        self.global_steps = 0
        self.total_training_steps = 0
        self.workers: Dict[str, Any] = {}
        self.agent_group_worker: Dict[int, Dict[NodeRole, Any]] = defaultdict(dict)
        self.agent_group_process_group: Dict[int, Dict[NodeRole, Any]] = defaultdict(dict)
        self.process_groups: Dict[str, ProcessGroup] = {}
        self.tokenizer_mapping: Dict[str, TokenizerModule] = {}
        self.kl_ctrl_in_reward = None
        self.logger = None
        self.progress_bar = None
        self._rank: int = -1
        self.taskgraph: Optional[TaskGraph] = None
        self.internal_data_cache: Dict[str, DataProto] = {}
        self.agent_critic_worker: Any
        # Finish flag
        self.taskgraph_execute_finished = False

        # async rollout
        self.rollout_mode = "sync"
        self._async_rollout_manager = None
        self.zmq_address = None  # used for async_vllmrollout

        # Add a cache to hold data from an insufficient batch for the next training step.
        # This is the core state-carrying mechanism for dynamic sampling.
        self.sampling_leftover_cache: Optional[DataProto] = None

        # multi agent
        self._multi_agent = False
        try:
            self._initialize_worker()
        except (ValueError, TypeError, KeyError, AttributeError, NotImplementedError) as e:
            rank = os.environ.get("RANK", "UNKNOWN")
            logger.error(f"Rank {rank}: Failed to create DAGWorker due to a critical setup error: {e}", exc_info=True)
            raise DAGInitializationError(f"Initialization failed on Rank {rank}: {e}") from e

        self.log_ray_actor_info()
