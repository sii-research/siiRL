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
import uuid
import ray
import time
import torch
import asyncio
import psutil
import random
import numpy as np
import torch.distributed as dist
from collections import defaultdict
from pprint import pformat
from tqdm import tqdm
from loguru import logger
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from torch.distributed import ProcessGroup

from siirl.models.loader import TokenizerModule, load_tokenizer
from siirl.global_config.params import SiiRLArguments
from siirl.engine.base_worker import Worker
from siirl.execution.dag import TaskGraph
from siirl.execution.dag.node import NodeRole, NodeType, Node
from siirl.execution.scheduler.reward import compute_reward, create_reward_manager
from siirl.execution.scheduler.process_group_manager import ProcessGroupManager
from siirl.execution.scheduler.enums import AdvantageEstimator
from siirl.data_coordinator import DataProto
from siirl.data_coordinator.dataloader import DataLoaderNode
from siirl.data_coordinator.protocol import collate_fn
from siirl.dag_worker.data_structures import NodeOutput, ValidationPayload, ValidationResult
from siirl.dag_worker.constants import DAGConstants, DAGInitializationError
from siirl.dag_worker.core_algos import (
    agg_loss, 
    apply_kl_penalty, 
    compute_advantage, 
    compute_response_mask
    )
from siirl.dag_worker.metric_aggregator import (
    METRIC_CONFIG_FULL,
    METRIC_CONFIG_MEAN_ONLY,
    DistributedMetricAggregator,
    _ReduceOp
)
from siirl.dag_worker.dag_utils import  (
    remove_prefix_from_dataproto,
    add_prefix_to_dataproto,
    add_prefix_to_metrics,
    log_ray_actor_info,
    get_and_validate_rank,
    get_taskgraph_for_rank,
    log_role_worker_mapping,
    should_create_worker,
    generate_node_worker_key,
    generate_agent_group_key,
    find_first_non_compute_ancestor,
    setup_sharding_manager,
    get_worker_classes,
    get_parallelism_config,
    prepare_generation_batch,
    prepare_generation_batch, 
    dump_validation_generations,
    whether_put_data,
    format_metrics_by_group,
    log_metrics_to_console,
    timer,
    consistent_hash,
    log_core_performance_metrics,
    aggregate_and_write_performance_metrics,
    prepare_local_batch_metrics,
    )
from siirl.utils.debug import DistProfiler
from siirl.utils.extras.device import get_device_name, get_nccl_backend, get_device_id
from siirl.utils.metrics.metric_utils import (
    aggregate_validation_metrics, 
    compute_throughout_metrics, 
    compute_timing_metrics
    )
from siirl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from siirl.execution.rollout_flow.multiturn.agent_loop import AgentLoopManager
from siirl.dag_worker.checkpoint_manager import CheckpointManager

device_name = get_device_name()

class DAGWorker(Worker):
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

        log_ray_actor_info(self._rank)

# ==========================================================================================
# Module 1: Execution and Training Loop
# ==========================================================================================

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
                        self.checkpoint_manager.save_checkpoint(self.global_steps)

                    # (Logging and validation logic remains unchanged)
                    metrics_dict = dict(ordered_metrics)
                    if self.val_reward_fn and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        val_metrics = self._validate()
                        if self._rank == 0 and val_metrics:
                            metrics_dict.update(val_metrics)
                        if is_last_step:
                            last_val_metrics = val_metrics

                    if self.enable_perf:
                        aggregate_and_write_performance_metrics(self._gather_group, self._rank, self.global_steps, self.config, metrics_dict)

                    ordered_metric_dict = format_metrics_by_group(metrics_dict, DAGConstants.METRIC_GROUP_ORDER)
                    log_core_performance_metrics(self._rank, self.enable_perf, ordered_metric_dict, self.global_steps)
                    if self._rank == 0:
                        if self.logger:
                            self.logger.log(data=ordered_metric_dict, step=self.global_steps)
                        else:
                            log_metrics_to_console(self._rank, ordered_metric_dict, self.global_steps)

                if self.progress_bar and not (epoch == start_epoch and batch_idx < batches_to_skip):
                    self.progress_bar.update(1)

        if self._rank == 0 and last_val_metrics:
            logger.info(f"Final validation metrics:\n{pformat(last_val_metrics)}")

    def _cleanup_step_buffers(self, visited_nodes: Set[str], timing_raw: dict) -> None:
        """
        Encapsulates the logic for resetting and clearing all step-related buffers.
        This includes the distributed Ray data buffers and the local internal cache.
        This is called at the end of a step, whether it completed successfully or was aborted.
        """
        # Reset the distributed (Ray) buffers for all keys that were used in this step.
        with timer(self.enable_perf, "reset_data_buffer", timing_raw):
            self.reset_data_buffer(list(visited_nodes))
        # Clear the local, in-process cache for the next step.
        with timer(self.enable_perf, "reset_intern_data_buffer", timing_raw):
            self.internal_data_cache.clear()

    def _run_training_step(self, epoch: int, batch_idx: int) -> Optional[List[Tuple[str, Any]]]:
        """Executes a single training step by traversing the computational graph."""
        timing_raw, ordered_metrics = {}, []

        with timer(self.enable_perf, "step", timing_raw):
            # --- 1. Data Loading ---
            with timer(self.enable_perf, "data_loading", timing_raw):
                batch = DataProto.from_single_dict(self.dataloader.run(epoch=epoch, is_validation_step=False))

            with timer(self.enable_perf, "get_entry_node", timing_raw):
                node_queue = self.taskgraph.get_entry_nodes()
                if not node_queue:
                    logger.error("Task graph has no entry nodes. Cannot start execution.")
                    return None

                entry_node_id = node_queue[0].node_id

            # --- 2. Graph Traversal ---
            visited_nodes = set()
            with timer(self.enable_perf, "graph_execution", timing_raw):
                while node_queue:
                    with timer(self.enable_perf, "graph_loop_management", timing_raw):
                        cur_node = node_queue.pop(0)
                        if cur_node.node_id in visited_nodes:
                            continue
                        visited_nodes.add(cur_node.node_id)

                        cur_dp_size, cur_dp_rank, cur_tp_rank, cur_tp_size, cur_pp_rank, cur_pp_size = self._get_node_dp_info(cur_node)
                        logger.debug(f"current node({cur_node.node_id}) dp_size: {cur_dp_size}, dp_rank: {cur_dp_rank}, tp_rank: {cur_tp_rank}, pp_rank: {cur_pp_rank}, pp_size: {cur_pp_size}")
                    from siirl.execution.dag.node import NodeRole
                
                    # --- 3. Get Input Data ---
                    if cur_node.node_id != entry_node_id:
                        with timer(self.enable_perf, "get_data_from_buffer", timing_raw):
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
                        with timer(self.enable_perf, "get_data_from_buffer_barrier", timing_raw):
                            dist.barrier(self._gather_group)
                    # --- 4. Node Execution ---

                    node_name_timer = f"{cur_node.node_role.name.lower()}"
                    if cur_node.only_forward_compute and cur_node.node_role == NodeRole.ACTOR:
                        node_name_timer = "actor_log_prob"
                    with timer(self.enable_perf, node_name_timer, timing_raw):
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
                        with timer(self.enable_perf, f"{node_name_timer}_barrier", timing_raw):
                            dist.barrier(self._gather_group)
                    if cur_node.node_role == NodeRole.ROLLOUT and self._multi_agent:
                        next_nodes = self.taskgraph.get_downstream_nodes(cur_node.node_id)
                        while next_nodes[0].node_role == NodeRole.ROLLOUT:
                            cur_node = next_nodes[0]
                            next_nodes = self.taskgraph.get_downstream_nodes(cur_node.node_id)
                            
                    # --- 5. Process Output & Pass to Children ---
                    with timer(self.enable_perf, "graph_output_handling", timing_raw):
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
                            if whether_put_data(self._rank, is_current_last_pp_tp_rank0, next_dp_size, cur_dp_size, cur_node, next_node):
                                with timer(self.enable_perf, "put_data_to_buffer", timing_raw):
                                    if self._multi_agent and next_node.node_role == NodeRole.ADVANTAGE:
                                        self.multi_agent_put_log(key=next_node.node_id, data=node_output.batch, next_dp_size = next_dp_size, agent_group = next_node.agent_group, timing_raw = timing_raw)
                                    else:
                                        enforce_buffer = (self._multi_agent) and (cur_node.node_role == NodeRole.ADVANTAGE)
                                        self.put_data_to_buffers(key=next_node.node_id, data=node_output.batch, source_dp_size=cur_dp_size, dest_dp_size=next_dp_size, enforce_buffer = enforce_buffer,timing_raw=timing_raw)
                        elif self._multi_agent:
                            # last_node add prefix for metrics
                            node_output.batch = add_prefix_to_dataproto(node_output.batch, cur_node) 
                        if self.enable_perf:
                            with timer(self.enable_perf, "put_data_to_buffer_barrier", timing_raw):
                                dist.barrier(self._gather_group)
                        with timer(self.enable_perf, "get_next_node", timing_raw):
                            # Add unvisited downstream nodes to the queue
                            for n in next_nodes:
                                if n.node_id not in visited_nodes:
                                    node_queue.append(n)

                    # barrier after each node execution ensures synchronization.
                    # This is safer but might be slower. Can be configured to be optional.
                    with timer(self.enable_perf, "step_barrier", timing_raw):
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

# ==========================================================================================
# Module 2: Graph Node Execution Handlers
# ==========================================================================================

    def _set_node_executables(self):
        """Maps node roles to their corresponding execution methods."""
        ROLE_METHOD_MAPPING = {
            (NodeRole.ROLLOUT, False): self.generate,
            (NodeRole.REFERENCE, False): self.compute_ref_log_prob,
            (NodeRole.ACTOR, True): self.compute_old_log_prob,
            (NodeRole.ACTOR, False): self.train_actor,
            (NodeRole.CRITIC, True): self.compute_value,
            (NodeRole.CRITIC, False): self.train_critic,
        }
        for node in self.taskgraph.nodes.values():
            if node.node_role in [NodeRole.REWARD, NodeRole.ADVANTAGE]:
                continue
            key = (node.node_role, node.only_forward_compute)
            if executable_func := ROLE_METHOD_MAPPING.get(key):
                node.executable = executable_func

    @DistProfiler.annotate(role="generate")
    def generate_sync_mode(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Sync mode"""
        gen_batch:DataProto = prepare_generation_batch(batch)
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
            gen_batch = prepare_generation_batch(batch)
            loop = asyncio.get_event_loop()
            gen_output = loop.run_until_complete(self._async_rollout_manager.generate_sequences(gen_batch))
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
        gen_batch = prepare_generation_batch(batch)
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

# ==========================================================================================
# Module 3: Worker and Environment Initialization
# ==========================================================================================

    def _initialize_worker(self):
        """Orchestrates the ordered initialization of all worker components."""
        self._rank = get_and_validate_rank()
        self.taskgraph = get_taskgraph_for_rank(self._rank, self.taskgraph_mapping)
        
        self._setup_distributed_environment()
        self._setup_tokenizers()
        self._setup_dataloader_and_reward()
        self._setup_role_worker_mapping()
        self._initialize_node_workers()
        self._profiler = DistProfiler(rank=self._rank, config=self.config.profiler)

        # Initialize CheckpointManager - Note: will be fully initialized after workers are created
        self.checkpoint_manager = None

        if self._rank == 0:
            logger.info("Rank 0: Initializing tracking logger...")
            from siirl.utils.logger.tracking import Tracking

            self.logger = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=self.config.to_dict(),
            )
            if self.enable_perf:
                logger.warning("Performance tracking is enabled. This may impact training speed.")

    def _setup_distributed_environment(self):
        """Initializes the default process group and all required subgroups."""
        # gloo_socket_ifname = 'bond0'
        # os.environ["GLOO_SOCKET_IFNAME"] = gloo_socket_ifname
        # os.environ["GLOO_LOG_LEVEL"] = "DEBUG"
        
        if not dist.is_initialized():
            backend = (
                f"{get_nccl_backend()}"
                if self.world_size >= self.config.dag.backend_threshold
                else f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}"
            )
            logger.info(
                f"Rank {self._rank}: Initializing world size {self.world_size} default process group with '{backend}' "
                f"backend."
            )
            dist.init_process_group(backend=backend)

        if device_name == "npu":
            # For NPU, metrics aggregation requires the hccl backend for device-to-device communication.
            # This group is created regardless of world size for NPU environments.
            gather_backend = get_nccl_backend()
            self._gather_group = dist.new_group(backend=gather_backend)
        else:
            # For GPU, the original logic is preserved for backward compatibility.
            # The gather group is only created if world_size < backend_threshold.
            self._gather_group = dist.new_group(
                backend="gloo") if self.world_size < self.config.dag.backend_threshold else None
            
        group_specs = self.process_group_manager.get_all_specs()
        if not group_specs:
            logger.warning("No process group specifications found in ProcessGroupManager.")
            return

        #Builds all process groups defined in the ProcessGroupManager.
        for name, spec in group_specs.items():
            if not isinstance(spec, dict) or not (ranks := spec.get("ranks")):
                logger.warning(f"Skipping group '{name}' due to invalid spec or missing 'ranks'.")
                continue
            self.process_groups[name] = dist.new_group(ranks=ranks)
        logger.debug(f"Rank {self._rank}: Created {len(self.process_groups)} custom process groups.")
        
        self.inference_group_name_set = self.process_group_manager.get_process_group_for_node_type_in_subgraph(
            self.taskgraph.graph_id, NodeType.MODEL_INFERENCE.value
        )
        self.train_group_name_set = self.process_group_manager.get_process_group_for_node_type_in_subgraph(
            self.taskgraph.graph_id, NodeType.MODEL_TRAIN.value
        )

        # Ensure all ranks have finished group creation before proceeding.
        dist.barrier(self._gather_group)
        logger.info(f"Rank {self._rank}: Distributed environment setup complete.")

    def _setup_tokenizers(self):
        """Initializes and caches tokenizers for all models in the task graph."""
        model_nodes = [
            node
            for node in self.taskgraph.nodes.values()
            if node.node_type in [NodeType.MODEL_TRAIN, NodeType.MODEL_INFERENCE]
        ]
        if not model_nodes:
            logger.warning("No model nodes found in the task graph. Tokenizer setup will be skipped.")
            return

        for node in model_nodes:
            agent_key = generate_agent_group_key(node)
            if agent_key not in self.tokenizer_mapping:
                # Add robust check for missing configuration.
                intern_config = node.config.get(DAGConstants.INTERN_CONFIG)
                if not intern_config or not (model_dict := getattr(intern_config, "model", None)):
                    logger.warning(f"Node {node.node_id} is missing model config. Skipping tokenizer setup for it.")
                    continue

                tokenizer_module = load_tokenizer(model_args=model_dict)
                if tokenizer := tokenizer_module.get("tokenizer"):
                    tokenizer.padding_side = "left"  # Required for most causal LM generation
                self.tokenizer_mapping[agent_key] = tokenizer_module
        logger.info(f"Rank {self._rank}: Initialized {len(self.tokenizer_mapping)} tokenizer(s).")

    def _setup_dataloader_and_reward(self):
        """Initializes the data loader and reward functions."""
        rollout_nodes = [n for n in self.taskgraph.nodes.values() if n.node_type == NodeType.MODEL_INFERENCE]
        if not rollout_nodes:
            raise ValueError("At least one MODEL_INFERENCE node is required for dataloader and reward setup.")
        self.first_rollout_node = rollout_nodes[0]

        pg_assignment = self.process_group_manager.get_node_assignment(self.first_rollout_node.node_id)
        if not (process_group_name := pg_assignment.get("process_group_name")):
            raise ValueError(
                f"Process group name not found for the first rollout node {self.first_rollout_node.node_id}."
            )

        self.dataloader_process_group = self.process_groups.get(process_group_name)
        if self.dataloader_process_group is None:
            raise ValueError(f"Could not find process group '{process_group_name}' in the created groups.")

        self.dataloader_tensor_model_parallel_size = self.first_rollout_node.config[
            DAGConstants.INTERN_CONFIG
        ].rollout.tensor_model_parallel_size

        self.dataloader = DataLoaderNode(
            node_id="dataloader",
            global_config=self.config,
            config={
                "group_world_size": dist.get_world_size(self.dataloader_process_group),
                "group_rank": dist.get_rank(self.dataloader_process_group),
                "group_parallel_size": self.dataloader_tensor_model_parallel_size,
                "num_loader_workers": self.config.data.num_loader_workers,
                "auto_repeat": self.config.data.auto_repeat,
            },
        )

        self.validate_tokenizer = next(iter(self.tokenizer_mapping.values()), {}).get("tokenizer")
        if not self.validate_tokenizer:
            logger.warning("No tokenizer loaded; reward functions might fail or use a default one.")

        self.val_reward_fn = create_reward_manager(
            self.config,
            self.validate_tokenizer,
            num_examine=1,
            max_resp_len=self.config.data.max_response_length,
            overlong_buffer_cfg=self.config.reward_model.overlong_buffer,
        )
        self.reward_fn = create_reward_manager(
            self.config,
            self.validate_tokenizer,
            num_examine=0,
            max_resp_len=self.config.data.max_response_length,
            overlong_buffer_cfg=self.config.reward_model.overlong_buffer,
            **self.config.reward_model.reward_kwargs,
        )

        if self.config.algorithm.use_kl_in_reward:
            from siirl.dag_worker import core_algos
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

    def _setup_role_worker_mapping(self):
        """Creates a mapping from NodeRole to the corresponding Worker implementation class."""
        self.role_worker_mapping: Dict[NodeRole, Type[Worker]] = {}
        # Actor/Ref/Rollout/Critic workers
        actor_strategy = self.config.actor_rollout_ref.actor.strategy
        self.role_worker_mapping.update(get_worker_classes(self.config, actor_strategy))

        # Reward model worker (if enabled)
        if self.config.reward_model.enable:
            reward_strategy = self.config.reward_model.strategy
            reward_workers = get_worker_classes(self.config, reward_strategy)
            if NodeRole.REWARD in reward_workers:
                self.role_worker_mapping[NodeRole.REWARD] = reward_workers[NodeRole.REWARD]
            else:
                logger.warning(
                    f"Reward model is enabled, but no worker found for role REWARD with strategy {reward_strategy}."
                )

        log_role_worker_mapping(self.role_worker_mapping)

    def _initialize_node_workers(self):
        """Instantiates worker objects for all nodes in the task graph."""
        for node in self.taskgraph.nodes.values():
            if not should_create_worker(self.role_worker_mapping, node):
                continue

            worker_cls = self.role_worker_mapping.get(node.node_role)
            if not worker_cls:
                logger.warning(f"No worker class found for role {node.node_role.name}. Skipping node {node.node_id}.")
                continue

            node_worker_key = generate_node_worker_key(node)
            if node_worker_key in self.workers:
                continue

            try:
                node_process_group = self._get_node_process_group(node)
                config = node.config.get(DAGConstants.INTERN_CONFIG)
                if hasattr(config, "actor") and hasattr(config.actor, "optim"):
                    config.actor.optim.total_training_steps = self.dataloader.total_training_steps
                elif hasattr(config, "optim"):
                    config.optim.total_training_steps = self.dataloader.total_training_steps
                worker_args = {"config": config, "process_group": node_process_group}
               
                # For separated workers (Megatron backend), no role parameter is needed
                # Only legacy ActorRolloutRefWorker needs the role parameter
                if hasattr(worker_cls, '__name__') and 'ActorRolloutRefWorker' in worker_cls.__name__:
                    if node.node_role in DAGConstants.WORKER_ROLE_MAPPING:
                        worker_args["role"] = DAGConstants.WORKER_ROLE_MAPPING[node.node_role]
                if node.agent_options and node.agent_options.share_instance:
                    # cur agent share same critic with target agent
                    self.agent_group_worker[node.agent_group][node.node_role] = self.agent_group_worker[node.agent_options.share_instance][node.node_role]
                else:
                    worker_instance = worker_cls(**worker_args)
                    self.workers[node_worker_key] = worker_instance
                    self.agent_group_worker[node.agent_group][node.node_role] = worker_instance
                    self.agent_group_process_group[node.agent_group][node.node_role] = node_process_group
                    logger.success(
                        f"Rank {self._rank}: Successfully created worker '{worker_cls.__name__}' for node: {node.node_id}"
                    )

            except Exception as e:
                #  Explicitly log the failing node and worker class, then re-raise
                # the exception to prevent silent failures.
                logger.error(
                    f"Failed to create worker for node {node.node_id} with class {worker_cls.__name__}.", exc_info=True
                )
                raise RuntimeError(f"Worker instantiation failed for node {node.node_id}") from e
        
        if len(self.agent_group_worker) > 1:
            self._multi_agent = True

    def _get_node_process_group(self, node: Node) -> ProcessGroup:
        """Retrieves the PyTorch ProcessGroup assigned to a specific graph node."""
        assignment = self.process_group_manager.get_node_assignment(node.node_id)
        if not (assignment and (name := assignment.get("process_group_name"))):
            raise ValueError(f"Process group assignment or name not found for node {node.node_id}.")

        pg = self.process_groups.get(name)
        if pg is None:
            raise ValueError(f"Process group '{name}' for node {node.node_id} was not created or found.")
        return pg

    def _get_node(self, role: NodeRole, agent_group: int) -> Node:
        """
        Finds and returns a specific node from the task graph based on its role
        and agent group.
        """
        found_node = next(
            (
                node
                for node in self.taskgraph.nodes.values()
                if node.node_role == role and node.agent_group == agent_group
            ),
            None,
        )

        if found_node is None:
            raise RuntimeError(f"Could not find a node with role {role.name} for agent_group {agent_group}")
        return found_node

    def _get_node_dp_info(self, node: Node) -> tuple[int, int, int, int, int, int]:
        """
        Calculates Data Parallel (DP), Tensor Parallel (TP), and Pipeline Parallel (PP) info for a node.
        
        Returns:
            tuple: (dp_size, dp_rank, tp_rank, tp_size, pp_rank, pp_size)
        """
        reference_node = node
        if node.node_type == NodeType.COMPUTE:
            # If the node is a COMPUTE type, find its true data source ancestor.
            ancestor = find_first_non_compute_ancestor(self.taskgraph, node.node_id)
            if ancestor:
                reference_node = ancestor
            else:
                # If no non-COMPUTE ancestor is found, it's a critical error.
                raise RuntimeError(f"Could not find any non-COMPUTE ancestor for COMPUTE node '{node.node_id}'. Please check your DAG graph configuration.")

        if reference_node.node_type == NodeType.COMPUTE:
            group_world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            group_rank = dist.get_rank()
        else:
            process_group = self._get_node_process_group(reference_node)
            group_world_size = dist.get_world_size(process_group)
            group_rank = dist.get_rank(process_group)

        # Get parallelism configuration based on backend strategy
        tp_size, pp_size = get_parallelism_config(reference_node)
        
        # Calculate total parallel size (TP * PP)
        total_parallel_size = tp_size * pp_size
        
        if group_world_size % total_parallel_size != 0:
            raise ValueError(f"Configuration error for node {node.node_id}: Group world size ({group_world_size}) is not divisible by total parallel size (TP={tp_size} * PP={pp_size} = {total_parallel_size}). Check your parallel configuration.")
        
        dp_size = group_world_size // total_parallel_size
        
        # Calculate ranks within the data parallel group
        dp_rank = group_rank // total_parallel_size
        
        # Calculate position within the TP-PP grid
        local_rank_in_tp_pp_group = group_rank % total_parallel_size
        
        # For 2D parallelism: ranks are arranged as [PP0_TP0, PP0_TP1, ..., PP0_TP(tp_size-1), PP1_TP0, ...]
        pp_rank = local_rank_in_tp_pp_group // tp_size
        tp_rank = local_rank_in_tp_pp_group % tp_size
        
        return dp_size, dp_rank, tp_rank, tp_size, pp_rank, pp_size
    
    def init_model(self):
        """Initializes models for all workers and sets up sharding managers where applicable."""
        logger.info("Initializing models for all worker nodes...")
        have_init_workers = set()
        for node in self.taskgraph.nodes.values():
            if should_create_worker(self.role_worker_mapping, node):
                node_worker = self.workers[generate_node_worker_key(node)]
                if not isinstance(node_worker, Worker):
                    raise TypeError(f"Invalid worker type for node {node.node_id}: {type(node_worker).__name__}")
                if generate_node_worker_key(node) in have_init_workers:
                    logger.warning(
                        f"Rank {self._rank}: Worker {generate_node_worker_key(node)} for node {node.node_id} "
                        f"already initialized. Skipping."
                    )
                    continue
                node_worker.init_model()
                have_init_workers.add(generate_node_worker_key(node))
        logger.success("All worker models initialized.")

        logger.info(f"Setting up sharding managers {self.config.actor_rollout_ref.rollout.name} ...")
        for agent_group, worker_dict in self.agent_group_worker.items():
            if NodeRole.ACTOR in worker_dict and NodeRole.ROLLOUT in worker_dict:
                try:
                    setup_sharding_manager(self.config, self.agent_group_process_group, agent_group, worker_dict)
                except Exception as e:
                    logger.error(f"Failed to set up sharding manager for agent group {agent_group}: {e}", exc_info=True)
                    raise
                
        
        if self.config.actor_rollout_ref.rollout.mode == "async":
            logger.info(f"Initial Async Rollout Server ...")
            for node in self.taskgraph.nodes.values():
                if node.node_role == NodeRole.ROLLOUT:
                    # need init after set sharding manager
                    self.rollout_mode = "async"
                    node_worker = self.workers[generate_node_worker_key(node)]
                    self.zmq_address = node_worker.get_zeromq_address()
                    self.init_async_server(node=node, node_worker=node_worker)
        logger.info("All models and sharding managers initialized successfully.")
        if self._multi_agent:
            from siirl.execution.rollout_flow.multi_agent.multiagent_generate import MultiAgentLoop
            self.multi_agent_loop =  MultiAgentLoop(self, config = self.config.actor_rollout_ref, node_workers = self.workers, local_dag = self.taskgraph, databuffer = self.data_buffers, placement_mode = 'colocate')

    def init_graph(self):
        # this is needed by async rollout manager
        self._set_node_executables()
        self.init_model()

        # Initialize CheckpointManager after workers are created
        self.checkpoint_manager = CheckpointManager(
            config=self.config,
            rank=self._rank,
            gather_group=self._gather_group,
            workers=self.workers,
            taskgraph=self.taskgraph,
            dataloader=self.dataloader,
            first_rollout_node=self.first_rollout_node,
            get_node_dp_info_fn=self._get_node_dp_info
        )

        # Use CheckpointManager to load checkpoint
        self.global_steps = self.checkpoint_manager.load_checkpoint()

        # Ensure all models are initialized and checkpoints are loaded before starting.
        dist.barrier(self._gather_group)

    def init_async_server(self, node:Node, node_worker):
        #gather zmq_address to rank_0
        _, dp_rank, tp_rank, tp_size, *_ = self._get_node_dp_info(node)
        addr_len = len(self.zmq_address) 
        encoded_addr = torch.tensor([ord(c) for c in self.zmq_address], dtype=torch.uint8,
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        zmq_addresses = []
        if tp_rank == 0: 
            group_addrs = torch.zeros((tp_size, addr_len), dtype=torch.uint8, device=encoded_addr.device)
            group_addrs[0] = encoded_addr
            for i in range(1, tp_size):
                src_rank = dp_rank * tp_size + i
                dist.recv(group_addrs[i], src=src_rank)
            for i in range(tp_size):
                addr_str = ''.join([chr(c.item()) for c in group_addrs[i]])
                zmq_addresses.append(addr_str)
        else:  # 组内其他进程（tp1, tp3...）
            # 发送自己的地址到组内目标进程
            dist.send(encoded_addr, dst=dp_rank * tp_size)
        if tp_rank == 0:
            self._async_rollout_manager = AgentLoopManager(node.config["intern_config"], dp_rank, os.environ['WG_PREFIX'], node_worker.rollout, zmq_addresses)
    def get_zeromq_address(self):
        return self.zmq_address
    
# ==========================================================================================
# Module 4: Validation
# ==========================================================================================

    def _validate(self) -> Dict[str, float]:
        """Performs validation by generating, scoring, and aggregating metrics across all ranks."""
        self.val_timedict = defaultdict(float)
        if self._rank == 0:
            logger.info("=" * 60)
            logger.info(f"Starting Validation @ Global Step {self.global_steps}...")
            logger.info("=" * 60)
            self.val_timedict["overall_start_time"] = time.perf_counter()

        all_scored_results: List[ValidationResult] = []

        # Check if num_val_batches > 0 to avoid unnecessary loops.
        if self.dataloader.num_val_batches <= 0:
            if self._rank == 0:
                logger.warning("num_val_batches is 0. Skipping validation.")
            return {}

        for i in range(self.dataloader.num_val_batches):
            if self._rank == 0:
                logger.debug(f"Processing validation batch {i + 1}/{self.dataloader.num_val_batches}")

            with timer(self.enable_perf, "prep_and_generate", self.val_timedict):
                batch_proto = self._prepare_validation_batch()
                generated_proto = self._generate_for_validation(batch_proto)
                dist.barrier(self._gather_group)  

            with timer(self.enable_perf, "score_and_package", self.val_timedict):
                scored_results = self._score_and_package_results(generated_proto)
                all_scored_results.extend(scored_results)

        dump_validation_generations(self.config, self.global_steps, self._rank, all_scored_results)
        dist.barrier(self._gather_group)
        
        _, _, tp_rank, _, pp_rank, _ = self._get_node_dp_info(self.first_rollout_node)
        # Gather all payloads to rank 0
        with timer(self.enable_perf, "gather_payloads", self.val_timedict):
            payloads_for_metrics = []
            if tp_rank == 0 and pp_rank == 0:
                # Only the master rank of the TP group (tp_rank=0) and first PP stage (pp_rank=0) prepares the payload.
                payloads_for_metrics = [
                    ValidationPayload(r.input_text, r.score, r.data_source, r.extra_rewards) for r in all_scored_results
                ]
            gathered_payloads_on_rank0 = [None] * self.world_size if self._rank == 0 else None
            dist.gather_object(payloads_for_metrics, gathered_payloads_on_rank0, dst=0, group=self._gather_group)

        # Rank 0 performs the final aggregation and logging
        if self._rank == 0:
            flat_payload_list = [p for sublist in gathered_payloads_on_rank0 if sublist for p in sublist]
            final_metrics = self._aggregate_and_log_validation_metrics(flat_payload_list)
        dist.barrier(self._gather_group)
        
        return final_metrics if self._rank == 0 else {}

    def _prepare_validation_batch(self) -> DataProto:
        """Fetches and prepares a single batch for validation."""
        test_batch = self.dataloader.run(is_validation_step=True)
        test_batch_proto = DataProto.from_single_dict(test_batch)
        n_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
        return test_batch_proto.repeat(n_samples, interleave=True)

    def _generate_for_validation(self, batch_proto: DataProto) -> DataProto:
        """Generates sequences using the rollout worker for a validation batch."""
        rollout_worker = self.agent_group_worker[0][NodeRole.ROLLOUT]
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs

        prompt_texts = self.validate_tokenizer.batch_decode(
            batch_proto.batch["input_ids"], skip_special_tokens=True
        )
        batch_proto.non_tensor_batch["prompt_texts"] = prompt_texts

        gen_batch = prepare_generation_batch(batch_proto)

        if self.config.actor_rollout_ref.rollout.agent.rewards_with_env and "reward_model" in batch_proto.non_tensor_batch:
            gen_batch.non_tensor_batch["reward_model"] = batch_proto.non_tensor_batch["reward_model"] 
        gen_batch.meta_info = {
            "eos_token_id": self.validate_tokenizer.eos_token_id,
            "pad_token_id": self.validate_tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": val_kwargs.do_sample,
            "validate": True,
        }
        logger.info(f"_generate_for_validation gen batch meta_info: {gen_batch.meta_info}")
        
        output = None
        if self._multi_agent is False:
            if self.rollout_mode == 'sync':
                output = rollout_worker.generate_sequences(gen_batch)
            elif self._async_rollout_manager:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(self._async_rollout_manager.generate_sequences(gen_batch))
        else:
            output = self.multi_agent_loop.generate_sequence(gen_batch)
            if output:
                return output
            return batch_proto
        if output:
            batch_proto.union(output)
        return batch_proto

    def _score_and_package_results(self, generated_proto: DataProto) -> List[ValidationResult]:
        """Scores generated sequences and packages them into ValidationResult objects."""
        if self.rollout_mode == 'async' and self._async_rollout_manager is None:
            return []
        if self._multi_agent and 'responses' not in generated_proto.batch:
            return []
        if "token_level_rewards" in generated_proto.batch:
            reward_result = {"reward_tensor": generated_proto.batch["token_level_rewards"],
                             "reward_extra_info": {}}
        else:    
            reward_result = self.val_reward_fn(generated_proto, return_dict=True)
        scores = reward_result["reward_tensor"].sum(-1).cpu()

        input_texts = generated_proto.non_tensor_batch.get("prompt_texts")
        if input_texts is None:
            logger.error(
                "FATAL: `prompt_texts` not found in `non_tensor_batch`. "
                "The prompt data was lost during the process. Falling back to decoding the full sequence, "
                "but please be aware the resulting `input_text` will be INCORRECT (it will contain prompt + response)."
            )
            # Fallback to prevent a crash, but the output is known to be wrong.
            input_texts = self.validate_tokenizer.batch_decode(
                generated_proto.batch["input_ids"], skip_special_tokens=True
            )

        output_texts = self.validate_tokenizer.batch_decode(generated_proto.batch["responses"], skip_special_tokens=True)
        data_sources = generated_proto.non_tensor_batch.get("data_source", ["unknown"] * len(scores))
        extra_info = generated_proto.non_tensor_batch.get("extra_info", [None] * len(scores))

        packaged_results = []
        for i in range(len(scores)):
            if self.dataloader.is_val_trailing_rank and isinstance(extra_info[i], dict) and extra_info[i].get("padded_duplicate", None):
                logger.debug(f"Rank {self._rank} skip append padded duplicate item {i}: score={scores[i].item()}")
                continue
            extra_rewards = {k: v[i] for k, v in reward_result.get("reward_extra_info", {}).items()}
            packaged_results.append(ValidationResult(input_texts[i], output_texts[i], scores[i].item(), data_sources[i], reward_result["reward_tensor"][i], extra_rewards))
        return packaged_results

    def _aggregate_and_log_validation_metrics(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
        """On Rank 0, aggregates all validation results and logs performance."""
        if not all_payloads:
            logger.warning("Validation finished with no results gathered on Rank 0 to aggregate.")
            return {}

        logger.info(f"Rank 0: Aggregating {len(all_payloads)} validation results...")
        with timer(self.enable_perf, "final_aggregation", self.val_timedict):
            final_metrics = self._aggregate_validation_results(all_payloads)

        # Log performance breakdown
        total_time = time.perf_counter() - self.val_timedict.pop("overall_start_time", time.perf_counter())
        logger.info("--- Validation Performance Breakdown (Rank 0) ---")
        for name, duration in self.val_timedict.items():
            logger.info(f"  Total {name.replace('_', ' ').title():<25}: {duration:.4f}s")
        known_time = sum(self.val_timedict.values())
        logger.info(f"  {'Other/Overhead':<25}: {max(0, total_time - known_time):.4f}s")
        logger.info(f"  {'TOTAL VALIDATION TIME':<25}: {total_time:.4f}s")
        logger.info("=" * 51)

        return final_metrics

    def _aggregate_validation_results(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
        """Computes the final metric dictionary from all gathered validation payloads."""
        data_sources = [p.data_source for p in all_payloads]
        sample_inputs = [p.input_text for p in all_payloads]

        infos_dict = defaultdict(list)
        for p in all_payloads:
            infos_dict["reward"].append(p.score)
            for key, value in p.extra_rewards.items():
                infos_dict[key].append(value)

        data_src2var2metric2val = aggregate_validation_metrics(data_sources=data_sources, sample_inputs=sample_inputs, infos_dict=infos_dict)

        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                if not metric2val:
                    continue

                # Robustly parse '@N' to prevent crashes from malformed metric names.
                n_max_values = []
                for name in metric2val.keys():
                    if "@" in name and "/mean" in name:
                        try:
                            n_val = int(name.split("@")[-1].split("/")[0])
                            n_max_values.append(n_val)
                        except (ValueError, IndexError):
                            continue  # Ignore malformed metric names

                n_max = max(n_max_values) if n_max_values else 1

                for metric_name, metric_val in metric2val.items():
                    is_core_metric = (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name)

                    metric_sec = "val-core" if is_core_metric else "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Re-calculate test_score per data source

        data_source_rewards = defaultdict(list)
        for p in all_payloads:
            data_source_rewards[p.data_source].append(p.score)

        for source, rewards in data_source_rewards.items():
            if rewards:
                metric_dict[f"val/test_score/{source}"] = np.mean(rewards)

        return metric_dict

# ==========================================================================================
# Module 5: Utilities
# ==========================================================================================

    def put_data_to_buffers(
        self, key: str, 
        data: DataProto,
        source_dp_size: int, 
        dest_dp_size: int, 
        enforce_buffer: bool, 
        timing_raw: Dict[str, float]):
        """Puts data into shared Ray plasma store for consumption by downstream nodes."""
        try:
            logger.debug(f"Rank {self._rank}: Starting put_data_to_buffers for key '{key}', source_dp_size={source_dp_size}, dest_dp_size={dest_dp_size}")
            
            data.meta_info["padding_values"] = {"input_ids": self.validate_tokenizer.pad_token_id, "responses": self.validate_tokenizer.pad_token_id, "labels": -100, "attention_mask": 0, "response_mask": 0}
            data.meta_info["padding_side"] = self.validate_tokenizer.padding_side

            if (not enforce_buffer) and source_dp_size == dest_dp_size:
                with timer(self.enable_perf, f"put_intern_data_{key}", timing_raw):
                    logger.debug(f"Rank {self._rank}: DP size match ({source_dp_size}). Storing data for key '{key}' in local cache.")
                    self.internal_data_cache[key] = data
                    logger.debug(f"Rank {self._rank}: Successfully stored data for key '{key}' in local cache.")
            else:
                logger.debug(f"Rank {self._rank}: DP size mismatch (source={source_dp_size}, dest={dest_dp_size}). Using Ray buffers for key '{key}'.")
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    logger.debug(f"Rank {self._rank}: Creating new event loop for key '{key}'")
                    loop = asyncio.new_event_loop()

                with timer(self.enable_perf, f"put_ray_proto_data_{key}", timing_raw):
                    chunks = data.chunk(chunks=len(self.data_buffers))
                    logger.debug(f"Rank {self._rank}: Created {len(chunks)} chunks for key '{key}'")
                    put_futures = [buf.put.remote(key, chunk) for buf, chunk in zip(self.data_buffers, chunks)]
                
                with timer(self.enable_perf, f"put_proto_data_{key}", timing_raw):
                    try:
                        loop.run_until_complete(asyncio.gather(*put_futures))
                        logger.debug(f"Rank {self._rank}: Successfully stored all chunks for key '{key}' in Ray buffers")
                    except Exception as e:
                        logger.error(f"Rank {self._rank}: Failed to store chunks for key '{key}' in Ray buffers: {e}")
                        raise
        except Exception as e:
            logger.error(f"Rank {self._rank}: Unexpected error in put_data_to_buffers for key '{key}': {e}")
            raise  # Re-raise the exception to maintain the original behavior

    def get_data_from_buffers(
        self, 
        key: str, 
        my_current_dp_rank: int, 
        my_current_dp_size: int, 
        timing_raw: Dict[str, float]
        ) -> Optional[DataProto]:
        """Gets data from shared buffers that was produced by an upstream node."""
        try:
            # First, check the high-speed internal cache.
            with timer(self.enable_perf, f"get_intern_data_{key}", timing_raw):
                if key in self.internal_data_cache:
                    logger.debug(f"Rank {self._rank}: Found data for key '{key}' in local cache. Bypassing Ray.")
                    return self.internal_data_cache.pop(key)

            # If not in the local cache, fall back to remote Ray buffers.
            logger.debug(f"Rank {self._rank}: Data for key '{key}' not in local cache. Fetching from remote buffers.")
            if not self.data_buffers:
                logger.error(f"Rank {self._rank}: data_buffers is None, cannot get data for key '{key}'")
                return None

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                logger.debug(f"Rank {self._rank}: Creating new event loop for key '{key}'")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            try:
                logger.debug(f"Rank {self._rank}: Attempting Ray remote call for key '{key}'")
                first_item = loop.run_until_complete(self.data_buffers[0].get.remote(key, my_current_dp_rank, my_current_dp_size))
                logger.debug(f"Rank {self._rank}: Completed Ray remote call for key '{key}', got result type: {type(first_item)}")
            except Exception as e:
                logger.error(f"Rank {self._rank}: Error getting data from Ray buffer for key '{key}': {e}")
                return None

            if first_item is None:
                logger.error(f"Rank {self._rank}: first_item is None for key '{key}'")
                return None

            if isinstance(first_item, ray.ObjectRef):
                with timer(self.enable_perf, f"get_ref_data_{key}", timing_raw):
                    try:
                        return loop.run_until_complete(first_item)
                    except Exception as e:
                        logger.error(f"Rank {self._rank}: Error resolving Ray ObjectRef for key '{key}': {e}")
                        return None
            elif isinstance(first_item, DataProto):
                try:
                    # If data was chunked, retrieve all chunks and concatenate
                    with timer(self.enable_perf, f"get_proto_data_{key}", timing_raw):
                        other_chunks_futures = [b.get.remote(key, my_current_dp_rank, my_current_dp_size) for b in self.data_buffers[1:]]
                        other_chunks = loop.run_until_complete(asyncio.gather(*other_chunks_futures))
                    with timer(self.enable_perf, f"get_proto_data_concat_chunks_{key}", timing_raw):
                        return DataProto.concat([first_item] + other_chunks)
                except Exception as e:
                    logger.error(f"Rank {self._rank}: Error concatenating chunks for key '{key}': {e}")
                    return None
            logger.error(f"Rank {self._rank}: first_item type {type(first_item)} is neither ray.ObjectRef nor DataProto for key '{key}'")
            return None

        except Exception as e:
            logger.error(f"Rank {self._rank}: Unexpected error in get_data_from_buffers for key '{key}': {e}")
            return None

    def reset_data_buffer(self, all_keys: List[str]):
        """
        Reset the data buffer for a given list of keys.
        """
        if self._rank == 0:
            loop = asyncio.get_event_loop()
            for data_buffer in self.data_buffers:
                loop.run_until_complete(data_buffer.reset.remote())

    def multi_agent_put_log(self, key: str, data: DataProto, agent_group: int, next_dp_size: int, timing_raw):
        def uuid_hex_to_bucket(uuid_hex: str, num_buckets: int = 8) -> int:
            return consistent_hash(uuid_hex) % num_buckets
        data_size = len(data)
        put_futures = []
        meta_info = data.meta_info
        with timer(self.enable_perf, f"put_ray_proto_data_{key}", timing_raw):
            for i in range(data_size):
                cur_data = data[i]
                request_id = cur_data.non_tensor_batch[f"agent_group_{agent_group}_request_id"]
                next_dp_rank = uuid_hex_to_bucket(request_id, next_dp_size)
                cur_key = key + f"_{next_dp_rank}"
                buf = random.choice(self.data_buffers)
                # slice of DataProto is DataProtoItem, will loss meta_info, need to recompute when use
                cur_data = collate_fn([cur_data])
                cur_data.meta_info = meta_info
                put_futures.append(buf.put.remote(cur_key, cur_data))
        with timer(self.enable_perf, f"put_proto_data_{key}", timing_raw):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.gather(*put_futures))

    def multi_agent_get_log(self, key: str, cur_dp_rank: int, agent_group: int, timing_raw):
        loop = asyncio.get_event_loop()
        key = key + f"_{cur_dp_rank}"
        prefix_key = f"agent_group_{agent_group}_"
        with timer(self.enable_perf, f"get_ref_data_{key}", timing_raw):
            tasks = [buf.get.remote(key) for buf in self.data_buffers]
            temp_data =loop.run_until_complete(asyncio.gather(*tasks))
            # temp_data = self.data_buffers.get(key) 
            datas = [item for t in temp_data if t is not None for item in t]
            sorted_datas = sorted(datas, key = lambda x : (x.non_tensor_batch[prefix_key + 'request_id'], -x.non_tensor_batch[prefix_key + 'traj_step']))
            meta_info = sorted_datas[0].meta_info
            dataproto = collate_fn(sorted_datas)
            dataproto.meta_info = meta_info
        return dataproto

    def check_spmd_mode(self):
        return self.rollout_mode == 'sync' and self._multi_agent == False

    def _reduce_and_broadcast_metrics(
        self, local_metrics: Dict[str, Union[float, List[float], torch.Tensor]], group: dist.ProcessGroup
    ) -> Dict[str, float]:
        """
        Aggregates metrics in a distributed environment using a dedicated helper class.
        For Pipeline Parallel setups, ensures all ranks have the same metric keys to avoid
        tensor shape mismatch during all_reduce operations.

        Args:
            local_metrics: A dictionary of metrics on each rank.
            group: The process group for the aggregation.

        Returns:
            A dictionary with the globally aggregated metrics, available on all ranks.
        """
        if not isinstance(local_metrics, dict) or not local_metrics:
            return {}

        world_size = dist.get_world_size(group)
        if world_size <= 1:
            # If not in a distributed setting, perform local aggregation only.
            aggregator = DistributedMetricAggregator(local_metrics, group=None)
            # The bucketed values are already the final values in a non-distributed case.
            final_metrics = {}
            for op_type, data in aggregator.op_buckets.items():
                for key, value in data:
                    if op_type == _ReduceOp.SUM: # value is a (sum, count) tuple
                        final_metrics[key] = value[0] / value[1] if value[1] > 0 else 0.0
                    else: # value is a float
                        final_metrics[key] = float(value)
            return final_metrics

        # In Megatron with Pipeline Parallel:
        # 1. First gather all metric keys from all ranks to ensure consistency
        local_keys = set(local_metrics.keys())
        all_keys_list = [None] * world_size
        dist.all_gather_object(all_keys_list, local_keys, group=group)
        
        # 2. Union all keys to get the complete set of expected metrics
        all_expected_keys = set()
        for keys_set in all_keys_list:
            all_expected_keys.update(keys_set)
        
        # 3. Use the aggregator with unified keys to perform communication
        aggregator = DistributedMetricAggregator(local_metrics, group)
        # NOTE(Ping Zhang): Ensure all ranks have the same metrics by adding missing ones with default values
        aggregator.op_buckets = aggregator._bucket_local_metrics(local_metrics, all_expected_keys)
        return aggregator.aggregate_and_get_results()

    def _collect_final_metrics(self, batch: DataProto, timing_raw: dict) -> Dict[str, float]:
        """
        Orchestrates the collection and computation of all metrics for a training step
        using a highly efficient, all_reduce-based aggregation strategy.

        This function replaces the old `compute -> reduce -> finalize` pipeline.
        """
        device_name = get_device_name()
        if device_name == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif device_name == "npu":
            torch.npu.reset_peak_memory_stats()

        final_metrics = {}

        # --- 1. Prepare all local metric data ---
        use_critic = any(node.node_role == NodeRole.CRITIC for node in self.taskgraph.nodes.values())
        local_data = prepare_local_batch_metrics(batch, use_critic=use_critic)

        # --- 2. Build the dictionary for our generic, high-performance aggregator ---
        # We want mean, max, and min for most standard metrics.
        metrics_to_aggregate = {}

        # Process metrics requiring mean, max, and min
        for key, prefix in METRIC_CONFIG_FULL.items():
            if key in local_data:
                # The aggregator determines the operation from the key.
                # We provide the same raw tensor for mean, max, and min calculations.
                metrics_to_aggregate[f"{prefix}/mean"] = local_data[key]
                metrics_to_aggregate[f"{prefix}_max"] = local_data[key]
                metrics_to_aggregate[f"{prefix}_min"] = local_data[key]

        # Process metrics requiring only mean
        for key, prefix in METRIC_CONFIG_MEAN_ONLY.items():
            if key in local_data:
                metrics_to_aggregate[f"{prefix}/mean"] = local_data[key]

        representative_actor_node = next(
            (n for n in self.taskgraph.nodes.values() if n.node_role == NodeRole.ACTOR), self.first_rollout_node
        )
        _, _, _, _, pp_rank_in_group, _ = self._get_node_dp_info(representative_actor_node)
        # (1) For TP: we have already taken TP into account when we set global_token_num in compute_reward.
        # see: siirl/workers/dag_worker/mixins/node_executors_mixin.py:compute_reward
        # (2) For PP: only PP rank 0 contributes to avoid double counting within PP groups
        # The aggregation will average across DP groups and multiply by world size to get global estimate
        if pp_rank_in_group == 0:
            local_token_sum = sum(batch.meta_info.get("global_token_num", [0]))
            metrics_to_aggregate["perf/total_num_tokens/mean"] = float(local_token_sum)

        # --- 3. Perform the aggregated, distributed reduction ---
        with timer(self.enable_perf, "metrics_aggregation", timing_raw):
            aggregated_metrics = self._reduce_and_broadcast_metrics(metrics_to_aggregate, self._gather_group)

        # Post-process keys and values for the final output
        for key, value in aggregated_metrics.items():
            if "_max" in key and "mem" not in key:
                final_metrics[key.replace("_max", "/max")] = value
            elif "_min" in key:
                final_metrics[key.replace("_min", "/min")] = value
            else:
                final_metrics[key] = value

        # Special handling for total_num_tokens to convert mean back to sum
        if "perf/total_num_tokens/mean" in final_metrics:
            final_metrics["perf/total_num_tokens"] = final_metrics.pop(
                "perf/total_num_tokens/mean"
            ) * dist.get_world_size(self._gather_group)

        # --- 4. Handle special cases like Explained Variance ---
        if use_critic:
            # Determine the correct device for distributed operations
            device_name = get_device_name()
            if device_name in ["cuda", "npu"]:
                device = f"{device_name}:{get_device_id()}"
            else:
                # Fallback to the device of an existing tensor. If it's CPU, all_reduce will fail,
                # which is the original problem, indicating a deeper issue.
                device = local_data["returns"].device
            # These components only need to be summed. We can do a direct all_reduce.
            components_to_sum = {k: v for k, v in local_data.items() if k.endswith("_comp")}
            for tensor in components_to_sum.values():
                if self._gather_group is not None:
                    dist.all_reduce(tensor.to(device), op=dist.ReduceOp.SUM, group=self._gather_group)

            # Now all ranks have the global sums and can compute the final value.
            N = local_data["returns"].numel()
            total_N_tensor = torch.tensor([N], dtype=torch.int64, device=local_data["returns"].device)
            if self._gather_group is not None:
                dist.all_reduce(total_N_tensor.to(device), op=dist.ReduceOp.SUM, group=self._gather_group)
            global_N = total_N_tensor.item()

            if global_N > 0:
                global_returns_sum = final_metrics["critic/returns/mean"] * global_N
                global_returns_sq_sum = components_to_sum["returns_sq_sum_comp"].item()
                global_error_sum = components_to_sum["error_sum_comp"].item()
                global_error_sq_sum = components_to_sum["error_sq_sum_comp"].item()

                mean_returns = global_returns_sum / global_N
                var_returns = (global_returns_sq_sum / global_N) - (mean_returns**2)

                mean_error = global_error_sum / global_N
                var_error = (global_error_sq_sum / global_N) - (mean_error**2)

                final_metrics["critic/vf_explained_var"] = 1.0 - var_error / (var_returns + 1e-8)
            else:
                final_metrics["critic/vf_explained_var"] = 0.0

        # --- 5. Add timing and other rank-0-only metrics ---
        # Only rank 0 needs to compute these for logging.
        if self._rank == 0:
            batch.meta_info["global_token_num"] = [final_metrics.get("perf/total_num_tokens", 0)]
            final_metrics.update(compute_throughout_metrics(batch, timing_raw, dist.get_world_size()))
            final_metrics["perf/process_cpu_mem_used_gb"] = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
            timing_metrics = compute_timing_metrics(batch, timing_raw)
            for key, value in timing_metrics.items():
                if key.startswith("timing_s/"):
                    final_metrics[key.replace("timing_s/", "perf/delta_time/")] = value

        # All ranks return the final metrics. Ranks other than 0 can use them if needed,
        # or just ignore them. This is cleaner than returning an empty dict.
        return final_metrics

    def _collect_multi_final_metrics(self, batch: DataProto, ordered_metrics: dict, timing_raw: dict) -> Dict[str, float]:
        node_queue = self.taskgraph.get_entry_nodes()
        visited_nodes = set()
        while node_queue:
            cur_node = node_queue.pop(0)
            if cur_node.node_id in visited_nodes:
                continue
            if cur_node.node_role !=  NodeRole.ROLLOUT:
                break
            batch = remove_prefix_from_dataproto(batch, cur_node)        
            final_metrics = self._collect_final_metrics(batch, timing_raw)
            final_metrics = add_prefix_to_metrics(final_metrics, cur_node)
            if final_metrics:
                ordered_metrics.extend(sorted(final_metrics.items()))
            if next_nodes := self.taskgraph.get_downstream_nodes(cur_node.node_id):
                for n in next_nodes:
                    if n.node_id not in visited_nodes:
                        node_queue.append(n)
            batch = add_prefix_to_dataproto(batch, cur_node)
        return ordered_metrics

