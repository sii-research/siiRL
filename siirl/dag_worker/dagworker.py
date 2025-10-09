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
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque
from pprint import pformat
from tqdm import tqdm

import ray
from loguru import logger
import torch.distributed as dist
from torch.distributed import ProcessGroup

from siirl.models.loader import TokenizerModule
from siirl.execution.scheduler.process_group_manager import ProcessGroupManager
from siirl.global_config.params import SiiRLArguments
from siirl.engine.base_worker import Worker
from siirl.execution.dag import TaskGraph
from siirl.execution.dag.node import NodeRole, NodeType, Node
from siirl.data_coordinator import DataProto
from siirl.dag_worker.dag_utils import  (
    remove_prefix_from_dataproto,
    add_prefix_to_dataproto,
    add_prefix_to_metrics
    )
from siirl.dag_worker.data_structures import NodeOutput


from .constants import DAGConstants, DAGInitializationError

from .mixins.execution_mixin import ExecutionMixin
from .mixins.initialization_mixin import InitializationMixin
from .mixins.node_executors_mixin import NodeExecutorsMixin
from .mixins.utilities_mixin import UtilitiesMixin
from .mixins.validation_mixin import ValidationMixin


class DAGWorker(InitializationMixin, NodeExecutorsMixin, ValidationMixin, UtilitiesMixin, Worker):
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
        
    def execute_task_graph(self):
        """Main entry point to start the DAG execution pipeline."""
        logger.info(f"Rank {self._rank}: Starting DAG execution pipeline...")
        logger.success(f"Rank {self._rank}: All components initialized. Starting training loop from step {self.global_steps + 1}.")

        if self.val_reward_fn and self.config.trainer.val_before_train:
            # _validate handles multi-rank logic internally
            val_metrics = self._validate()
            if self._rank == 0 and val_metrics and self.logger:
                logger.info(f"Initial validation metrics:\n{pformat(val_metrics)}")
                self.logger.log(data=val_metrics, step=self.global_steps)

            if self.config.trainer.val_only:
                logger.info("`val_only` is true. Halting after initial validation.")
                return
        self._run_training_loop()

        if self.progress_bar:
            self.progress_bar.close()
        self.taskgraph_execute_finished = True
        logger.success(f"Rank {self._rank}: DAG execution finished.")

    def _run_training_loop(self):
        """
        The main loop that iterates through training steps and epochs.
        """
        self.total_training_steps = self.dataloader.total_training_steps
        if self.dataloader.num_train_batches <= 0:
            if self._rank == 0:
                logger.warning(f"num_train_batches is {self.dataloader.num_train_batches}. The training loop will be skipped.")
            return

        if self._rank == 0:
            self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        last_val_metrics = None

        # Calculate starting epoch and batches to skip in that epoch for resumption.
        start_epoch = 0
        batches_to_skip = 0
        if self.dataloader.num_train_batches > 0:
            start_epoch = self.global_steps // self.dataloader.num_train_batches
            batches_to_skip = self.global_steps % self.dataloader.num_train_batches

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            for batch_idx in range(self.dataloader.num_train_batches):
                # If resuming, skip batches that have already been completed in the starting epoch.
                if epoch == start_epoch and batch_idx < batches_to_skip:
                    continue

                if self.global_steps >= self.total_training_steps:
                    logger.info(f"Rank {self._rank}: Reached total training steps. Exiting loop.")
                    if self._rank == 0 and last_val_metrics:
                        logger.info(f"Final validation metrics:\n{pformat(last_val_metrics)}")
                    return
                if self.global_steps in self.config.profiler.profile_steps:
                    self._profiler.start(role="e2e", profile_step=self.global_steps)
                ordered_metrics = self._run_training_step(epoch, batch_idx)
                if self.global_steps in self.config.profiler.profile_steps:
                    self._profiler.stop()

                if ordered_metrics is None:
                    if self.progress_bar:
                        self.progress_bar.update(1)
                    continue

                self.global_steps += 1

                if ordered_metrics is not None:
                    is_last_step = self.global_steps >= self.total_training_steps

                    # Save checkpoint at the configured frequency.
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        self._save_checkpoint()

                    # (Logging and validation logic remains unchanged)
                    metrics_dict = dict(ordered_metrics)
                    if self.val_reward_fn and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        val_metrics = self._validate()
                        if self._rank == 0 and val_metrics:
                            metrics_dict.update(val_metrics)
                        if is_last_step:
                            last_val_metrics = val_metrics

                    if self.enable_perf:
                        self._aggregate_and_write_performance_metrics(metrics_dict)

                    ordered_metric_dict = self.format_metrics_by_group(metrics_dict, DAGConstants.METRIC_GROUP_ORDER)
                    self._log_core_performance_metrics(ordered_metric_dict, self.global_steps)
                    if self._rank == 0:
                        if self.logger:
                            self.logger.log(data=ordered_metric_dict, step=self.global_steps)
                        else:
                            self._log_metrics_to_console(ordered_metric_dict, self.global_steps)

                if self.progress_bar and not (epoch == start_epoch and batch_idx < batches_to_skip):
                    self.progress_bar.update(1)

        if self._rank == 0 and last_val_metrics:
            logger.info(f"Final validation metrics:\n{pformat(last_val_metrics)}")

    def _find_first_non_compute_ancestor(self, start_node_id: str) -> Optional[Node]:
        """
        Traverses upwards from a starting node to find the first ancestor
        that is not of type COMPUTE.

        Uses a Breadth-First Search (BFS) strategy to prioritize finding the
        closest ancestor by level.
        """
        start_node = self.taskgraph.get_node(start_node_id)
        if not start_node:
            logger.warning(f"Could not find start node '{start_node_id}' in the graph.")
            return None

        if start_node.node_type != NodeType.COMPUTE:
            return start_node
        queue = deque(start_node.dependencies)
        visited = set(start_node.dependencies)
        node_id = start_node_id

        while queue:
            logger.debug(f"try find dependency node with ID '{node_id}' during upward search")
            node_id = queue.popleft()
            node = self.taskgraph.get_node(node_id)

            if not node:
                logger.warning(f"Could not find dependency node with ID '{node_id}' during upward search.")
                continue

            if node.node_type != NodeType.COMPUTE:
                return node

            for dep_id in node.dependencies:
                if dep_id not in visited:
                    visited.add(dep_id)
                    queue.append(dep_id)
        return None

    def _cleanup_step_buffers(self, visited_nodes: Set[str], timing_raw: dict) -> None:
        """
        Encapsulates the logic for resetting and clearing all step-related buffers.
        This includes the distributed Ray data buffers and the local internal cache.
        This is called at the end of a step, whether it completed successfully or was aborted.
        """
        # Reset the distributed (Ray) buffers for all keys that were used in this step.
        with self._timer("reset_data_buffer", timing_raw):
            self.reset_data_buffer(list(visited_nodes))
        # Clear the local, in-process cache for the next step.
        with self._timer("reset_intern_data_buffer", timing_raw):
            self.internal_data_cache.clear()

    def _run_training_step(self, epoch: int, batch_idx: int) -> Optional[List[Tuple[str, Any]]]:
        """Executes a single training step by traversing the computational graph."""
        timing_raw, ordered_metrics = {}, []

        with self._timer("step", timing_raw):
            # --- 1. Data Loading ---
            with self._timer("data_loading", timing_raw):
                batch = DataProto.from_single_dict(self.dataloader.run(epoch=epoch, is_validation_step=False))

            with self._timer("get_entry_node", timing_raw):
                node_queue = self.taskgraph.get_entry_nodes()
                if not node_queue:
                    logger.error("Task graph has no entry nodes. Cannot start execution.")
                    return None

                entry_node_id = node_queue[0].node_id

            # --- 2. Graph Traversal ---
            visited_nodes = set()
            with self._timer("graph_execution", timing_raw):
                while node_queue:
                    with self._timer("graph_loop_management", timing_raw):
                        cur_node = node_queue.pop(0)
                        if cur_node.node_id in visited_nodes:
                            continue
                        visited_nodes.add(cur_node.node_id)

                        cur_dp_size, cur_dp_rank, cur_tp_rank, cur_tp_size, cur_pp_rank, cur_pp_size = self._get_node_dp_info(cur_node)
                        logger.debug(f"current node({cur_node.node_id}) dp_size: {cur_dp_size}, dp_rank: {cur_dp_rank}, tp_rank: {cur_tp_rank}, pp_rank: {cur_pp_rank}, pp_size: {cur_pp_size}")
                    from siirl.execution.dag.node import NodeRole
                
                    # --- 3. Get Input Data ---
                    if cur_node.node_id != entry_node_id:
                        with self._timer("get_data_from_buffer", timing_raw):
                            if self._multi_agent and cur_node.node_role == NodeRole.ADVANTAGE:
                                batch = self.multi_agent_get_log(key=cur_node.node_id, cur_dp_rank = cur_dp_rank, agent_group = cur_node.agent_group,timing_raw = timing_raw)
                            else:
                                batch = self.get_data_from_buffers(key=cur_node.node_id, my_current_dp_rank=cur_dp_rank, my_current_dp_size=cur_dp_size, timing_raw=timing_raw)
                            if batch is None:
                                logger.error(f"Rank {self._rank}: Failed to get data for node {cur_node.node_id}. Skipping step.")
                                return None  # Abort the entire step
                            batch = remove_prefix_from_dataproto(batch, cur_node)
                            if batch is None:
                                logger.error(f"Rank {self._rank}: Failed to get data for node {cur_node.node_id}. Skipping step.")
                                return None  # Abort the entire step
                            logger.debug(f"current node({cur_node.node_id}) get data from databuffer batch size: {batch.batch.size()}")
                    if self.enable_perf:
                        with self._timer("get_data_from_buffer_barrier", timing_raw):
                            dist.barrier(self._gather_group)
                    # --- 4. Node Execution ---

                    node_name_timer = f"{cur_node.node_role.name.lower()}"
                    if cur_node.only_forward_compute and cur_node.node_role == NodeRole.ACTOR:
                        node_name_timer = "actor_log_prob"
                    with self._timer(node_name_timer, timing_raw):
                        if cur_node.node_role == NodeRole.REWARD:
                            if self.check_spmd_mode():
                                node_output = self.compute_reward(batch, cur_tp_size)
                            elif cur_tp_rank == 0:
                                node_output = self.compute_reward(batch, cur_tp_size)
                        elif cur_node.node_role == NodeRole.ADVANTAGE:
                            if self.check_spmd_mode():
                                node_output = self.compute_advantage(batch, cur_node = cur_node)
                            elif cur_tp_rank == 0:
                                node_output = self.compute_advantage(batch, cur_node = cur_node)
                        elif cur_node.executable:
                            if cur_node.agent_options and cur_node.agent_options.train_cycle:
                                cycle_round = self.global_steps // cur_node.agent_options.train_cycle
                                agent_num = len(self.agent_group_worker)
                                if cycle_round % agent_num == cur_node.agent_group:
                                    node_output = cur_node.run(batch=batch, worker_group_index=cur_node.agent_group, siirl_args=self.config)
                                else:
                                    node_output = NodeOutput(batch=batch)
                            else:
                                node_output = cur_node.run(batch=batch, worker_group_index=cur_node.agent_group, siirl_args=self.config)
                        else:  # Passthrough node
                            logger.warning(f"Node {cur_node.node_id} has no executable. Passing data through.")
                            node_output = NodeOutput(batch=batch)
                    if self.enable_perf:
                        with self._timer(f"{node_name_timer}_barrier", timing_raw):
                            dist.barrier(self._gather_group)
                    if cur_node.node_role == NodeRole.ROLLOUT and self._multi_agent:
                        next_nodes = self.taskgraph.get_downstream_nodes(cur_node.node_id)
                        while next_nodes[0].node_role == NodeRole.ROLLOUT:
                            cur_node = next_nodes[0]
                            next_nodes = self.taskgraph.get_downstream_nodes(cur_node.node_id)
                            
                    # --- 5. Process Output & Pass to Children ---
                    with self._timer("graph_output_handling", timing_raw):
                        if cur_node.node_role == NodeRole.POSTPROCESS_SAMPLING:
                            if len(node_output.batch) == 0:
                                logger.warning(f"Rank {self._rank}: Data after postprocess_sampling is insufficient. Caching and skipping the rest of the training step.")
                                self._cleanup_step_buffers(visited_nodes, timing_raw)
                                return None
                            if "postprocess_status" in node_output.metrics:
                                del node_output.metrics["postprocess_status"]

                        if self._rank == 0 and node_output.metrics:
                            if self._multi_agent:
                                node_output.metrics = add_prefix_to_metrics(node_output.metrics, cur_node)
                            ordered_metrics.extend(sorted(node_output.metrics.items()))
                        if next_nodes := self.taskgraph.get_downstream_nodes(cur_node.node_id):
                            # Currently supports single downstream node, can be extended to a loop.
                            next_node = next_nodes[0]
                            next_dp_size, _, _, _, _, _ = self._get_node_dp_info(next_node)
                            node_output.batch = add_prefix_to_dataproto(node_output.batch, cur_node)
                            is_current_last_pp_tp_rank0 = (cur_pp_rank == cur_pp_size - 1 and cur_tp_rank == 0)
                            if self._whether_put_data(is_current_last_pp_tp_rank0, next_dp_size, cur_dp_size, cur_node, next_node):
                                with self._timer("put_data_to_buffer", timing_raw):
                                    if self._multi_agent and next_node.node_role == NodeRole.ADVANTAGE:
                                        self.multi_agent_put_log(key=next_node.node_id, data=node_output.batch, next_dp_size = next_dp_size, agent_group = next_node.agent_group, timing_raw = timing_raw)
                                    else:
                                        enforce_buffer = (self._multi_agent) and (cur_node.node_role == NodeRole.ADVANTAGE)
                                        self.put_data_to_buffers(key=next_node.node_id, data=node_output.batch, source_dp_size=cur_dp_size, dest_dp_size=next_dp_size, enforce_buffer = enforce_buffer,timing_raw=timing_raw)
                        elif self._multi_agent:
                            # last_node add prefix for metrics
                            node_output.batch = add_prefix_to_dataproto(node_output.batch, cur_node) 
                        if self.enable_perf:
                            with self._timer("put_data_to_buffer_barrier", timing_raw):
                                dist.barrier(self._gather_group)
                        with self._timer("get_next_node", timing_raw):
                            # Add unvisited downstream nodes to the queue
                            for n in next_nodes:
                                if n.node_id not in visited_nodes:
                                    node_queue.append(n)

                    # barrier after each node execution ensures synchronization.
                    # This is safer but might be slower. Can be configured to be optional.
                    with self._timer("step_barrier", timing_raw):
                        dist.barrier(self._gather_group)

            # --- 6. Final Metrics Collection ---
            self._cleanup_step_buffers(visited_nodes, timing_raw)

        if self._multi_agent:
            ordered_metrics = self._collect_multi_final_metrics(batch, ordered_metrics, timing_raw)
        else:
            final_metrics = self._collect_final_metrics(batch, timing_raw)
            if final_metrics:
                ordered_metrics.extend(sorted(final_metrics.items()))

        ordered_metrics.extend([("training/global_step", self.global_steps + 1), ("training/epoch", epoch + 1)])
        return ordered_metrics

