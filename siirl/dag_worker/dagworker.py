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
import numpy as np
import torch
import torch.distributed as dist
from collections import defaultdict
from pprint import pformat
from tqdm import tqdm
from loguru import logger
from typing import Any, Dict, List, Optional, Set, Tuple, Type
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
from siirl.dag_worker.data_structures import NodeOutput
from siirl.dag_worker.constants import DAGConstants, DAGInitializationError
from siirl.dag_worker.core_algos import (
    agg_loss, 
    apply_kl_penalty, 
    compute_advantage, 
    compute_response_mask
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
    prepare_generation_batch
    )
from siirl.utils.debug import DistProfiler
from siirl.utils.extras.device import get_device_name, get_nccl_backend

from .mixins.utilities_mixin import UtilitiesMixin
from .mixins.validation_mixin import ValidationMixin

device_name = get_device_name()

class DAGWorker(ValidationMixin, UtilitiesMixin, Worker):
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

# ==========================================================================================
# Module 2: Graph Node Execution Handlers
# ==========================================================================================

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
                if node.node_role == NodeRole.ROLLOUT and node.config["intern_config"].rollout.mode == "async":
                    self.rollout_mode = "async"
                    self.zmq_address = node_worker.get_zeromq_address()
        logger.success("All worker models initialized.")

        logger.info(f"Setting up sharding managers {self.config.actor_rollout_ref.rollout.name} ...")
        for agent_group, worker_dict in self.agent_group_worker.items():
            if NodeRole.ACTOR in worker_dict and NodeRole.ROLLOUT in worker_dict:
                try:
                    setup_sharding_manager(self.config, self.agent_group_process_group, agent_group, worker_dict)
                except Exception as e:
                    logger.error(f"Failed to set up sharding manager for agent group {agent_group}: {e}", exc_info=True)
                    raise
        logger.info("All models and sharding managers initialized successfully.")
        if self._multi_agent:
            from siirl.execution.rollout_flow.multi_agent.multiagent_generate import MultiAgentLoop
            self.multi_agent_loop =  MultiAgentLoop(self, config = self.config.actor_rollout_ref, node_workers = self.workers, local_dag = self.taskgraph, databuffer = self.data_buffers, placement_mode = 'colocate')

    def init_graph(self):
        # this is needed by async rollout manager
        self._set_node_executables()
        self.init_model()
        self._load_checkpoint()
        # Ensure all models are initialized and checkpoints are loaded before starting.
        dist.barrier(self._gather_group)

    def set_async_rollout_manager(self, async_rollout_manager):
        self._async_rollout_manager = async_rollout_manager

    def get_zeromq_address(self):
        return self.zmq_address
