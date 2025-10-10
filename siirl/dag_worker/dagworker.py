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
import uuid
from tqdm import tqdm
import numpy as np

import ray
from loguru import logger
import torch
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
from siirl.utils.debug import DistProfiler
from siirl.dag_worker.core_algos import (
    agg_loss, 
    apply_kl_penalty, 
    compute_advantage, 
    compute_response_mask
    )
from siirl.execution.scheduler.reward import compute_reward
from siirl.execution.scheduler.enums import AdvantageEstimator

from .constants import DAGConstants, DAGInitializationError

from .mixins.initialization_mixin import InitializationMixin
from .mixins.utilities_mixin import UtilitiesMixin
from .mixins.validation_mixin import ValidationMixin


class DAGWorker(InitializationMixin, ValidationMixin, UtilitiesMixin, Worker):
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

    @DistProfiler.annotate(role="generate")
    def generate_sync_mode(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Sync mode"""
        gen_batch:DataProto = self._prepare_generation_batch(batch)
        if self.config.actor_rollout_ref.rollout.name == 'sglang':
            gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        gen_output = self.agent_group_worker[worker_group_index][NodeRole.ROLLOUT].generate_sequences(gen_batch)
        metrics = gen_output.meta_info.get("metrics", {})
        gen_output.meta_info = {}
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
        batch = batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True).union(gen_output)
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        return NodeOutput(batch=batch, metrics=metrics)

    @DistProfiler.annotate(role="generate")
    def generate_async_mode(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Async mode"""
        if self._async_rollout_manager is not None:
            gen_batch = self._prepare_generation_batch(batch)
            gen_output = self._async_rollout_manager.generate_sequences(gen_batch)
            metrics = gen_output.meta_info.get("metrics", {})
            gen_output.meta_info = {}
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
            batch = batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True).union(gen_output)
            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)
            return NodeOutput(batch=batch, metrics=metrics)
        return NodeOutput(batch=batch, metrics={})

    @DistProfiler.annotate(role="generate")
    def generate_multi_agent_mode(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Generates sequences for a training batch using the multi-agent rollout model."""
        gen_batch = self._prepare_generation_batch(batch)
        if self.config.actor_rollout_ref.rollout.agent.rewards_with_env and "reward_model" in batch.non_tensor_batch:
            gen_batch.non_tensor_batch["reward_model"] = batch.non_tensor_batch["reward_model"] 
        assert self.config.actor_rollout_ref.rollout.name == 'sglang'
        gen_output = self.multi_agent_loop.generate_sequence(gen_batch)
        if gen_output:
            metrics = gen_output.meta_info.get("metrics", {})
            # gen_output.meta_info = {}
            # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
            # batch = batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True).union(gen_output)
            # if "response_mask" not in batch.batch:
            #     batch.batch["response_mask"] = compute_response_mask(batch)
            return NodeOutput(batch=gen_output, metrics=metrics) 
        return NodeOutput(batch=batch, metrics={})

    @DistProfiler.annotate(role="generate")
    def generate(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Generates sequences for a training batch using the rollout model."""
        if self._multi_agent is False:
            if self.rollout_mode == 'sync':
                return self.generate_sync_mode(worker_group_index, batch, **kwargs)
            else: 
                return self.generate_async_mode(worker_group_index, batch, **kwargs)
        else:
            return self.generate_multi_agent_mode(worker_group_index, batch, **kwargs)
                    
    @DistProfiler.annotate(role="compute_reward")
    def compute_reward(self, batch: DataProto, tp_size: int, **kwargs) -> NodeOutput:
        """Calculates rewards for a batch of generated sequences."""
        if "token_level_rewards" in batch.batch:
            return NodeOutput(batch=batch, metrics={})
        batch.meta_info["global_token_num"] = (torch.sum(batch.batch["attention_mask"], dim=-1) // tp_size).tolist()
        reward_tensor, extra_infos = compute_reward(batch, self.reward_fn)
        batch.batch["token_level_scores"] = reward_tensor

        if extra_infos:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in extra_infos.items()})

        metrics = {}
        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl_in_reward, self.config.algorithm.kl_penalty)
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        return NodeOutput(batch=batch, metrics=metrics)

    @DistProfiler.annotate(role="compute_old_log_prob")
    def compute_old_log_prob(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Computes log probabilities from the actor model before the policy update."""
        if "global_token_num" not in batch.meta_info:
            # in multi-agent, agentA may don't have reward node
            # insert some info needed
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.ACTOR].compute_log_prob(batch)
        process_group = self._get_node_process_group(self._get_node(NodeRole.ACTOR, worker_group_index))

        local_metrics = processed_data.meta_info.get("metrics", {})
        if "entropys" in processed_data.batch:
            entropy = agg_loss(processed_data.batch["entropys"], processed_data.batch["response_mask"].to("cpu"), self.config.actor_rollout_ref.actor.loss_agg_mode)
            local_metrics["actor/entropy_loss"] = entropy.item()
        metrics = self._reduce_and_broadcast_metrics(local_metrics, process_group)

        processed_data.meta_info.pop("metrics", None)
        processed_data.batch.pop("entropys", None)

        if "rollout_log_probs" in processed_data.batch and self._rank == 0:
            rollout_probs, actor_probs = torch.exp(processed_data.batch["rollout_log_probs"]), torch.exp(processed_data.batch["old_log_probs"])
            rollout_probs_diff = torch.masked_select(torch.abs(rollout_probs.cpu() - actor_probs), processed_data.batch["response_mask"].bool().cpu())
            if rollout_probs_diff.numel() > 0:
                metrics.update({"training/rollout_probs_diff_max": torch.max(rollout_probs_diff).item(), "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).item(), "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).item()})
        return NodeOutput(batch=processed_data, metrics=metrics)

    @DistProfiler.annotate(role="compute_ref_log_prob")
    def compute_ref_log_prob(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Computes log probabilities from the frozen reference model."""
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.REFERENCE].compute_ref_log_prob(batch)
        metrics = processed_data.meta_info.get("metrics", {})
        return NodeOutput(batch=processed_data, metrics=metrics)

    @DistProfiler.annotate(role="compute_value")
    def compute_value(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Computes value estimates from the critic model."""
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.CRITIC].compute_values(batch)
        return NodeOutput(batch=processed_data)

    @DistProfiler.annotate(role="compute_advantage")
    def compute_multi_agent_advantage(self, batch: DataProto, **kwargs) -> NodeOutput:
        adv_config = self.config.algorithm
        rollout_config = self.config.actor_rollout_ref.rollout
        cur_node = kwargs["cur_node"]
        if "token_level_rewards" not in batch.batch :
            # make sure rewards of angentB has been compute
            # GAE_MARFT adv need make sure only last agent has adv node
            if depend_nodes := self.taskgraph.get_dependencies(cur_node.node_id):
                depend_node = depend_nodes[0]
                if adv_config.share_reward_in_agent:
                    batch.batch["token_level_rewards"] = batch.batch[f"agent_group_{depend_node.agent_group}_token_level_rewards"].clone()
                else:    
                    batch.batch["token_level_rewards"] = torch.zeros_like(batch.batch[f"agent_group_{depend_node.agent_group}_token_level_rewards"])
                batch.batch["token_level_scores"] = batch.batch[f"agent_group_{depend_node.agent_group}_token_level_scores"].clone()
            else:
                raise RuntimeError(f"cur_node {cur_node.node_id} have no rewards with can't find it's dependencies reward")
        if adv_config.adv_estimator == AdvantageEstimator.GAE_MARFT:
            # make sure adv node define in last agent node
            cur_agent_id = len(self.agent_group_worker) - 1
            agent_groups_ids = list(range(cur_agent_id))
            kwargs["agent_group_ids"] = agent_groups_ids
            # pre_agent may have no reward token
            for agent_id in reversed(agent_groups_ids):
                key_prefix = f"agent_group_{agent_id}_token_level_rewards"
                if key_prefix not in batch.batch:
                    pre_key_prefix = f"agent_group_{agent_id + 1}_token_level_rewards" if agent_id != cur_agent_id -1 else "token_level_rewards"
                    if adv_config.share_reward_in_agent:
                        batch.batch[key_prefix] = batch.batch[pre_key_prefix].clone()
                    else:
                        batch.batch[key_prefix] = torch.zeros_like(batch.batch[pre_key_prefix])
                batch.batch[f"agent_group_{agent_id}_token_level_scores"] = batch.batch[key_prefix].clone()
                    
        return NodeOutput(
            batch=compute_advantage(
                batch, adv_estimator=adv_config.adv_estimator, gamma=adv_config.gamma, lam=adv_config.lam, num_repeat=rollout_config.n, norm_adv_by_std_in_grpo=adv_config.norm_adv_by_std_in_grpo, weight_factor_in_cpgd=adv_config.weight_factor_in_cpgd, multi_turn=rollout_config.multi_turn.enable,
                **kwargs
            )
        )        
        
    @DistProfiler.annotate(role="compute_advantage")
    def compute_advantage(self, batch: DataProto, **kwargs) -> NodeOutput:
        """Computes advantages and returns for PPO using GAE."""
        if self._multi_agent:
            return self.compute_multi_agent_advantage(batch, **kwargs)
        adv_config = self.config.algorithm
        rollout_config = self.config.actor_rollout_ref.rollout
        return NodeOutput(
            batch=compute_advantage(
                batch, adv_estimator=adv_config.adv_estimator, gamma=adv_config.gamma, lam=adv_config.lam, num_repeat=rollout_config.n, norm_adv_by_std_in_grpo=adv_config.norm_adv_by_std_in_grpo, weight_factor_in_cpgd=adv_config.weight_factor_in_cpgd, multi_turn=rollout_config.multi_turn.enable,
                **kwargs
            )
        )

    @DistProfiler.annotate(role="train_critic")
    def train_critic(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Performs a single training step on the critic model."""
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.CRITIC].update_critic(batch)
        process_group = self._get_node_process_group(self._get_node(NodeRole.CRITIC, worker_group_index))
        metrics = self._reduce_and_broadcast_metrics(processed_data.meta_info.get("metrics"), process_group)
        return NodeOutput(batch=processed_data, metrics=metrics)

    @DistProfiler.annotate(role="train_actor")
    def train_actor(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Performs a single training step on the actor (policy) model."""
        if self.config.trainer.critic_warmup > self.global_steps:
            return NodeOutput(batch=batch)  # Skip actor update during critic warmup

        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.ACTOR].update_actor(batch)
        process_group = self._get_node_process_group(self._get_node(NodeRole.ACTOR, worker_group_index))
        metrics = self._reduce_and_broadcast_metrics(processed_data.meta_info.get("metrics"), process_group)
        return NodeOutput(batch=processed_data, metrics=metrics)
