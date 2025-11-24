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
import torch
import asyncio
import numpy as np
import torch.distributed as dist
from collections import defaultdict
from pprint import pformat
from tqdm import tqdm
from loguru import logger
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Callable
from torch.distributed import ProcessGroup
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
import time
from siirl.execution.metric_worker.metric_worker import MetricClient
from siirl.models.loader import TokenizerModule, load_tokenizer
from siirl.params import SiiRLArguments
from siirl.engine.base_worker import Worker
from siirl.execution.dag import TaskGraph
from siirl.execution.dag.node import NodeRole, NodeType, Node
from siirl.execution.scheduler.reward import compute_reward, create_reward_manager
from siirl.execution.scheduler.process_group_manager import ProcessGroupManager
from siirl.execution.scheduler.enums import AdvantageEstimator, WorkflowType
from siirl.data_coordinator import preprocess_dataloader, Samples2Dict, Dict2Samples, SampleInfo
from siirl.data_coordinator import DataProto
from siirl.data_coordinator.dataloader import DataLoaderNode
from siirl.dag_worker.data_structures import NodeOutput
from siirl.dag_worker.constants import DAGConstants, DAGInitializationError
from siirl.dag_worker import core_algos
from siirl.dag_worker.checkpoint_manager import CheckpointManager
from siirl.dag_worker.core_algos import (
    agg_loss,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask
    )
from siirl.dag_worker.dag_utils import  (
    log_ray_actor_info,
    get_and_validate_rank,
    get_taskgraph_for_rank,
    log_role_worker_mapping,
    should_create_worker,
    generate_node_worker_key,
    find_first_non_compute_ancestor,
    setup_sharding_manager,
    get_worker_classes,
    get_parallelism_config,
    prepare_generation_batch,
    format_metrics_by_group,
    log_metrics_to_console,
    aggregate_and_write_performance_metrics,
    log_core_performance_metrics,
    timer,
    reduce_and_broadcast_metrics,
    whether_put_data
    )
from siirl.utils.debug import DistProfiler
from siirl.utils.extras.device import get_device_name, get_nccl_backend
from siirl.execution.rollout_flow.multiturn.agent_loop import AgentLoopManager

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
        data_coordinator: "ray.actor.ActorHandle",
        metric_worker: "ray.actor.ActorHandle",
        device_name="cuda",
    ):
        super().__init__()
        self.config = config
        self.process_group_manager = process_group_manager
        self.taskgraph_mapping = taskgraph_mapping
        self.data_coordinator = data_coordinator
        self.device_name = device_name
        self.enable_perf = os.environ.get("SIIRL_ENABLE_PERF", "0") == "1" or config.dag.enable_perf

        # State attributes
        self.timing_raw = {}
        self.global_steps = 0
        self.total_training_steps = 0
        self.workers: Dict[str, Any] = {}
        self.multi_agent_group: Dict[int, Dict[NodeRole, Any]] = defaultdict(dict)
        self.agent_group_process_group: Dict[int, Dict[NodeRole, Any]] = defaultdict(dict)
        self.process_groups: Dict[str, ProcessGroup] = {}
        self.tokenizer_mapping: Dict[str, TokenizerModule] = {}
        self.logger = None
        self.progress_bar = None
        self._rank: int = -1
        self.taskgraph: Optional[TaskGraph] = None
        self.internal_data_cache: Dict[str, Any] = {}
        self.sample_ref_cache: list = []
        self.agent_critic_worker: Any
        # Finish flag
        self.taskgraph_execute_finished = False

        # async rollout
        self.rollout_mode = "sync"
        self._async_rollout_manager = None
        self.zmq_address = None  # used for async_vllmrollout

        # Add a cache to hold data from an insufficient batch for the next training step.
        # This is the core state-carrying mechanism for dynamic sampling.
        self.sampling_leftover_cache: Optional[Any] = None

        # multi agent
        self._multi_agent = False
        
        # metirc_worker
        self.metric_worker = MetricClient(metric_worker=metric_worker)
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

        if self.config.trainer.val_before_train:
            self.validator.validate(global_step=self.global_steps)
            self.metric_worker.wait_submit()
            dist.barrier(self._gather_group)
            if self._rank == 0 and self.logger:
                val_metrics = self.metric_worker.wait_final_res() 
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
            is_embodied = self.config.algorithm.workflow_type == WorkflowType.EMBODIED
            if is_embodied:
                self._cleanup_step_buffers(self.timing_raw)
            for batch_idx in range(self.dataloader.num_train_batches):
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

                is_last_step = self.global_steps >= self.total_training_steps

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    self.checkpoint_manager.save_checkpoint(self.global_steps)

                if self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    self.validator.validate(global_step=self.global_steps)
                    self.metric_worker.wait_submit()
                    dist.barrier(self._gather_group)
                    if self._rank == 0:
                        val_metric = self.metric_worker.wait_final_res()
                        ordered_metrics.update(val_metric)
                        if is_last_step:
                            last_val_metrics = val_metric

                if self.enable_perf:
                    aggregate_and_write_performance_metrics(self._gather_group, self._rank, self.global_steps, self.config, ordered_metrics)
                ordered_metric_dict = format_metrics_by_group(ordered_metrics, DAGConstants.METRIC_GROUP_ORDER)
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

    def _cleanup_step_buffers(self, timing_raw: dict) -> None:
        """
        Encapsulates the logic for resetting and clearing all step-related buffers.
        This includes the distributed Ray data buffers and the local internal cache.
        This is called at the end of a step, whether it completed successfully or was aborted.
        """
        # Reset the distributed (Ray) buffers for all keys that were used in this step.
        with timer(self.enable_perf, "reset_data_buffer", timing_raw):
            self.reset_data_buffer()
            for ref in self.sample_ref_cache:
                ray.internal.free(ref)
            self.sample_ref_cache = []
        # Clear the local, in-process cache for the next step.
        with timer(self.enable_perf, "reset_intern_data_buffer", timing_raw):
            self.internal_data_cache.clear()

    def _run_training_step(self, epoch: int, batch_idx: int) -> Optional[List[Tuple[str, Any]]]:
        """Executes a single training step by traversing the computational graph."""
        timing_raw, ordered_metrics = self.timing_raw, []

        with timer(self.enable_perf, "step", timing_raw):
            # --- 1. Data Loading ---
            with timer(self.enable_perf, "get_data_from_dataloader", timing_raw):
                batch = preprocess_dataloader(self.dataloader.run(epoch=epoch, is_validation_step=False), self.config.actor_rollout_ref.rollout.n)
            node_queue = self.taskgraph.get_entry_nodes()
            if not node_queue:
                logger.error("Taskgraph has no entry nodes. Cannot start execution.")
                return None
            entry_node_id = node_queue[0].node_id

            # --- 2. Graph Traversal ---
            visited_nodes = set()
            with timer(self.enable_perf, "graph_execution", timing_raw):
                while node_queue:
                    cur_node = node_queue.pop(0)
                    if cur_node.node_id in visited_nodes:
                        continue
                    visited_nodes.add(cur_node.node_id)

                    cur_dp_size, cur_dp_rank, cur_tp_rank, cur_tp_size, cur_pp_rank, cur_pp_size = self._get_node_dp_info(cur_node)
                    logger.debug(f"current node({cur_node.node_id}) dp_size: {cur_dp_size}, dp_rank: {cur_dp_rank}, tp_rank: {cur_tp_rank}, pp_rank: {cur_pp_rank}, pp_size: {cur_pp_size}")

                    # --- 3. Get Input Data ---
                    if cur_node.node_id != entry_node_id:
                        with timer(self.enable_perf, "get_data_from_buffer", timing_raw):
                            batch = self.get_data_from_buffers(key=cur_node.node_id, cur_dp_size=cur_dp_size, cur_dp_rank=cur_dp_rank, timing_raw=timing_raw)
                            if batch is None:
                                if self.config.algorithm.filter_groups.enable:
                                    if cur_node.node_role == NodeRole.ACTOR:
                                        logger.error(f"Rank {self._rank}: Failed to get data for node {cur_node.node_id}. Skipping step.")
                                        return None 
                                else:
                                    logger.error(f"Rank {self._rank}: Failed to get data for node {cur_node.node_id}. Skipping step.")
                                    return None 
                            else:
                                # batch = remove_prefix_from_dataproto(batch, cur_node)
                                logger.debug(f"current node({cur_node.node_id}) get data from databuffer batch size: {batch.size()}")
                    if self.enable_perf:
                        with timer(self.enable_perf, "get_data_from_buffer_barrier", timing_raw):
                            dist.barrier(self._gather_group)
                    # --- 4. Node Execution ---
                    node_name_timer = f"{cur_node.node_id}"
                    with timer(self.enable_perf, node_name_timer, timing_raw):
                        if cur_node.executable and batch is not None:
                            node_kwargs = {"_dag_worker_instance": self}
                            node_kwargs["process_group"] = self._get_node_process_group(cur_node) if cur_node.node_type != NodeType.COMPUTE else None
                            node_kwargs["agent_group"] = self.multi_agent_group[cur_node.agent_group]
                            node_kwargs["cur_tp_rank"] = cur_tp_rank
                            if cur_node.node_role == NodeRole.REWARD:
                                node_kwargs["tp_size"] = cur_tp_size
                            elif cur_node.node_role == NodeRole.ADVANTAGE:
                                node_kwargs["cur_node"] = cur_node

                            if cur_node.agent_options and cur_node.agent_options.train_cycle:
                                cycle_round = self.global_steps // cur_node.agent_options.train_cycle
                                agent_num = len(self.multi_agent_group)
                                if cycle_round % agent_num == cur_node.agent_group:
                                    node_output = cur_node.run(batch=batch,
                                                               config=self.config,
                                                               **node_kwargs)
                                else:
                                    node_output = NodeOutput(batch=batch)
                            else:
                                node_output = cur_node.run(batch=batch,
                                                           config=self.config,
                                                           **node_kwargs)
                        else:
                            logger.warning(f"Node {cur_node.node_id} has no executable. Passing data through.")
                            node_output = NodeOutput(batch=batch)
                    
                    # Check if node returned empty batch (e.g., DAPO insufficient samples)
                    # This triggers re-rollout to collect more data
                    if  node_output.batch is None or (node_output.batch is not None and len(node_output.batch) == 0):
                        logger.warning(
                            f"Rank {self._rank}: Node '{cur_node.node_id}' returned empty batch. "
                        )
                        if not self.config.algorithm.filter_groups.enable:
                            logger.warning(
                                f"Rank {self._rank}: Node '{cur_node.node_id}' returned empty batch. "
                                f"Aborting current step to trigger re-rollout. {node_output.batch is not None and len(node_output.batch) != 0}"
                            )
                            return None
                    
                    if self.enable_perf:        
                        with timer(self.enable_perf, f"{node_name_timer}_barrier", timing_raw):
                            dist.barrier(self._gather_group)
                    if cur_node.node_role == NodeRole.ROLLOUT and self._multi_agent:
                        next_nodes = self.taskgraph.get_downstream_nodes(cur_node.node_id)
                        while next_nodes[0].node_role == NodeRole.ROLLOUT:
                            cur_node = next_nodes[0]
                            next_nodes = self.taskgraph.get_downstream_nodes(cur_node.node_id)

                    # --- 5. Process Output & Get next node ---
                    with timer(self.enable_perf, "graph_output_handling", timing_raw):
                        if node_output.metrics and cur_tp_rank == 0 and cur_pp_rank == 0:
                            self.metric_worker.submit_metric(node_output.metrics, cur_dp_size)
                        if next_nodes := self.taskgraph.get_downstream_nodes(cur_node.node_id):
                            if node_output.batch is not None and len(node_output.batch) != 0:
                                # Currently supports single downstream node, can be extended to a loop.
                                next_node = next_nodes[0]
                                next_dp_size, _, _, _, _, _ = self._get_node_dp_info(next_node)
                                # node_output.batch = add_prefix_to_dataproto(node_output.batch, cur_node)
                                is_current_last_pp_tp_rank0 = (cur_pp_rank == cur_pp_size - 1 and cur_tp_rank == 0)
                                if whether_put_data(self._rank, is_current_last_pp_tp_rank0, next_dp_size, cur_dp_size, cur_node, next_node):
                                    with timer(self.enable_perf, "put_data_to_buffer", timing_raw):
                                        # if self._multi_agent and next_node.node_role == NodeRole.ADVANTAGE:
                                        #     self.multi_agent_put_log(key=next_node.node_id, data=node_output.batch, next_dp_size = next_dp_size, agent_group = next_node.agent_group, timing_raw = timing_raw)
                                        # else:
                                        # have filter, must use databuffer to rebalance
                                        enforce_buffer = (self.config.algorithm.filter_groups.enable) and (cur_node.node_type == NodeType.COMPUTE) and (next_node.node_type == NodeType.MODEL_TRAIN) 
                                        self.put_data_to_buffers(key=next_node.node_id, data=node_output.batch,  source_dp_size=cur_dp_size, dest_dp_size=next_dp_size, enforce_buffer = enforce_buffer, timing_raw=timing_raw)
                        # elif self._multi_agent:
                        #     # last_node add prefix for metrics
                        #     node_output.batch = add_prefix_to_dataproto(node_output.batch, cur_node)                        
                        if self.enable_perf:
                            with timer(self.enable_perf, "put_data_to_buffer_barrier", timing_raw):
                                dist.barrier(self._gather_group)
                        with timer(self.enable_perf, "get_next_node", timing_raw):
                            for n in next_nodes:
                                if n.node_id not in visited_nodes:
                                    node_queue.append(n)

                    with timer(self.enable_perf, "step_barrier", timing_raw):
                        dist.barrier(self._gather_group)

            # --- 6. Final Metrics Collection ---
            self._cleanup_step_buffers(timing_raw)

        ordered_metrics = {}
        if cur_tp_rank == 0 and cur_pp_rank == 0:
            self.metric_worker.compute_local_data_metric(batch, cur_dp_size)
            self.metric_worker.compute_local_throughout_metrics(batch, timing_raw, 1, cur_dp_size)
            if self._rank == 0:
                # only use rank0 time metrics
                self.metric_worker.compute_local_timing_metrics(batch, timing_raw, 1)  
        timing_raw.clear()
        self.metric_worker.wait_submit()
        dist.barrier(self._gather_group)
        if self._rank == 0:
            metrics = self.metric_worker.wait_final_res()
            ordered_metrics = dict(sorted(metrics.items()))
            ordered_metrics.update({"training/global_step": self.global_steps + 1, "training/epoch": epoch + 1})

        return ordered_metrics

# ==========================================================================================
# Module 2: Graph Node Execution Handlers
# ==========================================================================================

    @DistProfiler.annotate(role="generate")
    def generate_sync_mode(self, agent_group, batch: TensorDict) -> NodeOutput:
        """Sync mode"""
        gen_output = agent_group[NodeRole.ROLLOUT].generate_sequences(batch)
        if "response_mask" not in batch:
            gen_output["response_mask"] = compute_response_mask(gen_output)
        batch = batch.update(gen_output)
        return NodeOutput(batch=batch, metrics=gen_output["metrics"])

    @DistProfiler.annotate(role="generate")
    def generate_async_mode(self, batch: TensorDict) -> NodeOutput:
        """Async mode"""
        if self._async_rollout_manager is not None:
            loop = asyncio.get_event_loop()
            gen_output = loop.run_until_complete(self._async_rollout_manager.generate_sequences(batch))
            metrics = gen_output["metrics"]
            if "response_mask" not in batch:
                batch["response_mask"] = compute_response_mask(batch)
            return NodeOutput(batch=batch, metrics=metrics)
        return NodeOutput(batch=batch, metrics={})

    @DistProfiler.annotate(role="generate")
    def generate_multi_agent_mode(self, config, batch: DataProto) -> NodeOutput:
        """Generates sequences for a training batch using the multi-agent rollout model."""
        gen_batch = prepare_generation_batch(batch)
        if config.actor_rollout_ref.rollout.agent.rewards_with_env and "reward_model" in batch.non_tensor_batch:
            gen_batch.non_tensor_batch["reward_model"] = batch.non_tensor_batch["reward_model"]
        assert config.actor_rollout_ref.rollout.name == 'sglang'
        gen_output = self.multi_agent_loop.generate_sequence(gen_batch)
        if gen_output:
            metrics = gen_output.meta_info.get("metrics", {})
            # gen_output.meta_info = {}
            # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
            # batch = batch.repeat(config.actor_rollout_ref.rollout.n, interleave=True).union(gen_output)
            # if "response_mask" not in batch.batch:
            #     batch.batch["response_mask"] = compute_response_mask(batch)
            return NodeOutput(batch=gen_output, metrics=metrics)
        return NodeOutput(batch=batch, metrics={})

    @DistProfiler.annotate(role="generate")
    def generate_embodied_mode(self, agent_group, batch: TensorDict, **kwargs) -> NodeOutput:
        """
        Generates embodied episodes for training.
        
        This method follows the same pattern as _generate_for_embodied_validation in validation_mixin,
        but configured for training mode (do_sample=True, validate=False).
        
        For embodied tasks, the batch contains task metadata (task_id, trial_id, etc.) from the dataloader.
        The rollout worker interacts with the environment and generates all required data
        (input_ids, pixel_values, responses, etc.) during environment rollout.
        
        Unlike text generation, we do NOT call _prepare_generation_batch because:
        1. The input batch doesn't have text-generation keys (input_ids, attention_mask, etc.)
        2. These keys will be generated by the embodied rollout worker during env interaction
        """
        from loguru import logger
        
        rollout_worker = agent_group[NodeRole.ROLLOUT]
        
        # Set meta_info for embodied training
        batch["eos_token_id"] = NonTensorData(self.validate_tokenizer.eos_token_id if self.validate_tokenizer else None)
        batch["n_samples"] = NonTensorData(self.config.actor_rollout_ref.rollout.n)
        batch["pad_token_id"] = NonTensorData(self.validate_tokenizer.pad_token_id if self.validate_tokenizer else None)        
        logger.info(
            f"[Embodied Validation] Batch variables: "
            f"{batch.batch_size[0]}, "
            f"eos_token_id={batch['eos_token_id']}, "
            f"pad_token_id={batch['pad_token_id']}, "
            f"n_samples={batch['n_samples']}, "
        )
        # Generate embodied episodes
        gen_output = rollout_worker.generate_sequences(batch)
        metrics = gen_output["metrics"]
        batch.update(gen_output)
        # Add unique IDs for tracking
        # Compute response mask if not already present
        if "response_mask" not in batch:
            batch["response_mask"] = compute_response_mask(batch)
        
        return NodeOutput(batch=batch, metrics=metrics)
    
    
    
    @DistProfiler.annotate(role="generate")
    def generate(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
        """Generates sequences for a training batch using the rollout model."""
        # Check if this is embodied mode
        agent_group = kwargs.pop("agent_group")
        is_embodied = self.config.actor_rollout_ref.model.model_type == "embodied"
        
        if is_embodied:
            # Use dedicated embodied generation path (mirrors validation logic)
            return self.generate_embodied_mode(agent_group, batch, **kwargs)
        if self._multi_agent is False:
            if self.rollout_mode == 'sync':
                return self.generate_sync_mode(agent_group, batch)
            else:
                return self.generate_async_mode(batch)
        else:
            return self.generate_multi_agent_mode(config, batch)

    @DistProfiler.annotate(role="compute_reward")
    def compute_reward(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
        """Calculates rewards for a batch of generated sequences."""
        
        if not self.check_mode() and kwargs["cur_tp_rank"] != 0:
            return NodeOutput(batch=batch, metrics={})
        
        tp_size = kwargs.pop("tp_size")
        if "token_level_rewards" in batch and batch["token_level_rewards"].numel() > 0:
            return NodeOutput(batch=batch, metrics={})
        batch["global_token_num"] = NonTensorData((torch.sum(batch["attention_mask"], dim=-1) // tp_size).tolist())

        reward_tensor, extra_infos = compute_reward(batch, self.reward_fn)
        batch["token_level_scores"] = reward_tensor

        if extra_infos:
            batch.update({k: np.array(v) for k, v in extra_infos.items()}, inplace=True)

        metrics = {}
        if config.algorithm.use_kl_in_reward:
            kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl_in_reward, config.algorithm.kl_penalty)
            metrics.update(kl_metrics)
        else:
            batch["token_level_rewards"] = batch["token_level_scores"]
        return NodeOutput(batch=batch, metrics=metrics)

    
    @DistProfiler.annotate(role="compute_old_log_prob")
    def compute_old_log_prob(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
        """Computes log probabilities from the actor model before the policy update."""
        process_group = kwargs.pop("process_group")
        agent_group = kwargs.pop("agent_group")
        if "global_token_num" not in batch:
            # in multi-agent, agentA may don't have reward node
            # insert some info needed
            batch["global_token_num"] = NonTensorData(torch.sum(batch["attention_mask"], dim=-1).tolist())
        processed_data = agent_group[NodeRole.ACTOR].compute_log_prob(batch)
        local_metrics = processed_data["metrics"]  if "metrics" in processed_data else {}
        if "entropys" in processed_data:
            entropy = agg_loss(processed_data["entropys"], processed_data["response_mask"].to("cpu"), config.actor_rollout_ref.actor.loss_agg_mode)
            local_metrics["actor/entropy_loss"] = entropy.item()

        processed_data.pop("metrics", None)
        processed_data.pop("entropys", None)

        return NodeOutput(batch=processed_data, metrics=local_metrics)

    @DistProfiler.annotate(role="compute_ref_log_prob")
    def compute_ref_log_prob(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
        """Computes log probabilities from the frozen reference model."""
        agent_group = kwargs.pop("agent_group")
        processed_data = agent_group[NodeRole.REFERENCE].compute_ref_log_prob(batch)
        metrics = processed_data["metrics"]
        return NodeOutput(batch=processed_data, metrics=metrics)

    @DistProfiler.annotate(role="compute_value")
    def compute_value(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
        """Computes value estimates from the critic model."""
        agent_group = kwargs.pop("agent_group")
        processed_data = agent_group[NodeRole.CRITIC].compute_values(batch)
        return NodeOutput(batch=processed_data)

    @DistProfiler.annotate(role="compute_advantage")
    def compute_multi_agent_advantage(self, config, batch: DataProto, **kwargs) -> NodeOutput:
        adv_config = config.algorithm
        rollout_config = config.actor_rollout_ref.rollout
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
            cur_agent_id = len(self.multi_agent_group) - 1
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
                batch,
                adv_estimator=adv_config.adv_estimator,
                gamma=adv_config.gamma,
                lam=adv_config.lam,
                num_repeat=rollout_config.n,
                norm_adv_by_std_in_grpo=adv_config.norm_adv_by_std_in_grpo,
                weight_factor_in_cpgd=adv_config.weight_factor_in_cpgd,
                multi_turn=rollout_config.multi_turn.enable,
                **kwargs
            )
        )

    @DistProfiler.annotate(role="compute_advantage")
    def compute_advantage(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
        """Computes advantages and returns for PPO using GAE."""
        
        if not self.check_mode() and kwargs["cur_tp_rank"] != 0:
            return NodeOutput(batch=batch, metrics={})
        
        if self._multi_agent:
            return self.compute_multi_agent_advantage(config, batch, **kwargs)
        algo_config = config.algorithm
        return NodeOutput(
            batch=compute_advantage(
                batch,
                adv_estimator=algo_config.adv_estimator,
                gamma=algo_config.gamma,
                lam=algo_config.lam,
                norm_adv_by_std_in_grpo=algo_config.norm_adv_by_std_in_grpo,
                weight_factor_in_cpgd=algo_config.weight_factor_in_cpgd,
                **kwargs
            )
        )

    @DistProfiler.annotate(role="train_critic")
    def train_critic(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
        """Performs a single training step on the critic model."""
        agent_group = kwargs.pop("agent_group")
        process_group = kwargs.pop("process_group")
        processed_data = agent_group[NodeRole.CRITIC].update_critic(batch)
        return NodeOutput(batch=processed_data, metrics=processed_data["metrics"])

    @DistProfiler.annotate(role="train_actor")
    def train_actor(self, config, batch: TensorDict, **kwargs) -> NodeOutput:
        """Performs a single training step on the actor (policy) model."""
        process_group = kwargs.pop("process_group")
        agent_group = kwargs.pop("agent_group")
        global_steps = batch["global_steps"] if "global_steps" in batch else 0
        if config.trainer.critic_warmup > global_steps:
            return NodeOutput(batch=batch)  # Skip actor update during critic warmup
        batch["multi_turn"] = NonTensorData(self.config.actor_rollout_ref.rollout.multi_turn.enable)
        processed_data = agent_group[NodeRole.ACTOR].update_actor(batch)
        return NodeOutput(batch=processed_data, metrics=processed_data["metrics"])


# ==========================================================================================
# Module 3: Worker and Environment Initialization
# ==========================================================================================

    def _initialize_worker(self):
        """Orchestrates the ordered initialization of all worker components."""
        self._rank = get_and_validate_rank()
        self.taskgraph = get_taskgraph_for_rank(self._rank, self.taskgraph_mapping)

        self._setup_distributed_environment()
        self._setup_tokenizers()
        self._setup_dataloader()
        self._setup_reward_managers()
        self._setup_role_worker_mapping()
        self._initialize_node_workers()
        self._profiler = DistProfiler(rank=self._rank, config=self.config.profiler)

        # Initialize CheckpointManager - Note: will be fully initialized after workers are created
        self.checkpoint_manager = None

        # Initialize Validator - Note: will be initialized in init_graph() after all workers are ready
        self.validator = None

        # Initialize MetricsCollector - Note: will be initialized in init_graph() after all dependencies are ready
        self.metrics_collector = None

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
            agent_key = f"group_key_{node.agent_group}"
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

    def _setup_dataloader(self):
        """Initializes the data loader for training and validation."""
        rollout_nodes = [n for n in self.taskgraph.nodes.values() if n.node_type == NodeType.MODEL_INFERENCE]
        if not rollout_nodes:
            raise ValueError("At least one MODEL_INFERENCE node is required for dataloader setup.")
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
        logger.info(f"Rank {self._rank}: DataLoader initialized with {self.dataloader.total_training_steps} total training steps.")

    def _setup_reward_managers(self):
        """Initializes reward managers for training and validation."""
        self.validate_tokenizer = next(iter(self.tokenizer_mapping.values()), {}).get("tokenizer")
        if not self.validate_tokenizer:
            logger.warning("No tokenizer loaded; reward functions might fail or use a default one.")

        self.reward_fn = create_reward_manager(
            self.config,
            self.validate_tokenizer,
            num_examine=0,
            max_resp_len=self.config.data.max_response_length,
            overlong_buffer_cfg=self.config.reward_model.overlong_buffer,
            **self.config.reward_model.reward_kwargs,
        )
        logger.info(f"Rank {self._rank}: Reward managers initialized.")

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
                    self.multi_agent_group[node.agent_group][node.node_role] = self.multi_agent_group[node.agent_options.share_instance][node.node_role]
                else:
                    worker_instance = worker_cls(**worker_args)
                    self.workers[node_worker_key] = worker_instance
                    self.multi_agent_group[node.agent_group][node.node_role] = worker_instance
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

        if len(self.multi_agent_group) > 1:
            self._multi_agent = True

    def init_graph(self):
        """
        Initializes the computation graph by loading models and restoring checkpoint state.

        Executed after _initialize_worker() across all workers via Ray remote call.
        This method include:
        (1) model weight loading,
        (2) weight sharding_manager setup,
        (3) async/multi-agent init,
        (4) validator init,
        (5) metrics collector init,
        (6) checkpoint restoration
        """

        self._load_model_weights()

        self._setup_sharding_manager()

        self._setup_async_rollout()

        self._setup_multi_agent_loop()

        self._init_validator()

        self._init_metrics_collector()

        self._init_checkpoint_manager()
        self.global_steps = self.checkpoint_manager.load_checkpoint()

        dist.barrier(self._gather_group)

    def _load_model_weights(self):
        """Loads model weights to GPU for all node workers."""
        logger.info("Loading model weights for all worker nodes...")
        initialized_workers = set()

        for node in self.taskgraph.nodes.values():
            if not should_create_worker(self.role_worker_mapping, node):
                continue

            worker_key = generate_node_worker_key(node)
            if worker_key in initialized_workers:
                continue

            node_worker = self.workers[worker_key]
            if not isinstance(node_worker, Worker):
                raise TypeError(f"Invalid worker type for node {node.node_id}: {type(node_worker).__name__}")

            node_worker.init_model()
            initialized_workers.add(worker_key)

        logger.success("All model weights loaded successfully.")

    def _setup_sharding_manager(self):
        """Sets up sharding managers for actor-rollout weight synchronization."""
        logger.info(f"Setting up weight sharing infrastructure ({self.config.actor_rollout_ref.rollout.name})...")

        for agent_group, worker_dict in self.multi_agent_group.items():
            if NodeRole.ACTOR in worker_dict and NodeRole.ROLLOUT in worker_dict:
                try:
                    setup_sharding_manager(
                        self.config,
                        self.agent_group_process_group,
                        agent_group,
                        worker_dict
                    )
                except Exception as e:
                    logger.error(f"Failed to set up sharding manager for agent group {agent_group}: {e}", exc_info=True)
                    raise

        logger.success("Weight sharing infrastructure initialized.")

    def _setup_async_rollout(self):
        """Initializes async rollout server if configured."""
        if self.config.actor_rollout_ref.rollout.mode != "async":
            return

        logger.info("Initializing async rollout server...")
        for node in self.taskgraph.nodes.values():
            if node.node_role == NodeRole.ROLLOUT:
                self.rollout_mode = "async"
                node_worker = self.workers[generate_node_worker_key(node)]
                self.zmq_address = node_worker.get_zeromq_address()
                self.init_async_server(node=node, node_worker=node_worker)

        logger.success("Async rollout server initialized.")

    def _setup_multi_agent_loop(self):
        """Initializes multi-agent loop if in multi-agent mode."""
        if not self._multi_agent:
            return

        logger.info("Initializing multi-agent loop...")
        from siirl.execution.rollout_flow.multi_agent.multiagent_generate import MultiAgentLoop

        self.multi_agent_loop = MultiAgentLoop(
            self,
            config=self.config.actor_rollout_ref,
            node_workers=self.workers,
            local_dag=self.taskgraph,
            databuffer=self.data_buffers,
            placement_mode='colocate'
        )

        logger.success("Multi-agent loop initialized.")

    def _init_validator(self):
        """Initializes validator for validation workflow."""
        logger.info("Initializing validator...")
        from siirl.dag_worker.validator import Validator

        self.validator = Validator(
            config=self.config,
            dataloader=self.dataloader,
            validate_tokenizer=self.validate_tokenizer,
            multi_agent_group=self.multi_agent_group,
            rollout_mode=self.rollout_mode,
            async_rollout_manager=self._async_rollout_manager,
            multi_agent_loop=getattr(self, 'multi_agent_loop', None),
            multi_agent=self._multi_agent,
            rank=self._rank,
            world_size=self.world_size,
            gather_group=self._gather_group,
            first_rollout_node=self.first_rollout_node,
            get_node_dp_info_fn=self._get_node_dp_info,
            enable_perf=self.enable_perf,
            metric_worker=self.metric_worker
        )
        logger.success("Validator initialized.")

    def _init_metrics_collector(self):
        """Initializes metrics collector for training metrics aggregation."""
        logger.info("Initializing metrics collector...")
        # from siirl.dag_worker.metrics_collector import MetricsCollector
        # self.metric_worker.init()
        # self.metrics_collector = MetricsCollector(
        #     rank=self._rank,
        #     world_size=self.world_size,
        #     gather_group=self._gather_group,
        #     taskgraph=self.taskgraph,
        #     first_rollout_node=self.first_rollout_node,
        #     get_node_dp_info_fn=self._get_node_dp_info,
        #     multi_agent=self._multi_agent,
        #     enable_perf=self.enable_perf,
        # )
        # logger.success("Metrics collector initialized.")

    def _init_checkpoint_manager(self):
        """Initializes checkpoint manager for saving/loading training state."""
        logger.info("Initializing checkpoint manager...")
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
        else:
            dist.send(encoded_addr, dst=dp_rank * tp_size)
        if tp_rank == 0:
            self._async_rollout_manager = AgentLoopManager(node.config["intern_config"], dp_rank, os.environ['WG_PREFIX'], node_worker.rollout, zmq_addresses)

# ==========================================================================================
# Module 4: Utilities
# ==========================================================================================
    def put_data_to_buffers(
        self, key: str,
        data: TensorDict,
        source_dp_size:int,
        dest_dp_size: int,
        enforce_buffer: bool,
        timing_raw: Dict[str, float]
    ):
        """
        Puts data into the DataCoordinator by converting it into individual Samples.
        The data is tagged with a 'key' to be retrieved by the correct downstream node.
        """
        try:
            batch_size = len(data) if data is not None else 0

            if source_dp_size == dest_dp_size and not enforce_buffer:
                with timer(self.enable_perf, f"put_intern_data_{key}", timing_raw):
                    self.internal_data_cache[key] = data
            else:
                samples = Dict2Samples(data)
                if not samples:
                    logger.warning(f"Rank {self._rank}: DataProto for key '{key}' converted to 0 samples. Nothing to put.")
                    return

                with timer(self.enable_perf, f"put_samples_to_coordinator_{key}", timing_raw):
                    sample_infos = []
                    for sample in samples:
                        # Convert uid to string (handle tensor uid from postprocess_sampling)
                        uid_val = getattr(sample, 'uid', uuid.uuid4().int)
                        if isinstance(uid_val, torch.Tensor):
                            uid_str = str(int(uid_val.item()))
                        else:
                            uid_str = str(uid_val)
                        
                        sample_infos.append(SampleInfo(
                            sum_tokens=getattr(sample, 'sum_tokens', int(sample.attention_mask.sum())),
                            prompt_length=getattr(sample, 'prompt_length', 0),
                            response_length=getattr(sample, 'response_length', 0),
                            uid=uid_str,
                            dict_info={
                                'key': key,
                                'source_dp_size': source_dp_size  # Store source DP size
                            }
                        ))
                    
                    # Although ray.put is called multiple times, it is more efficient than remote actor calls.
                    # This is the main source of the remaining overhead, but it is necessary
                    # to maintain sample-level traceability in the DataCoordinator.
                    with timer(self.enable_perf, f"ray_put_samples_{key}", timing_raw):
                        sample_refs = [ray.put(sample) for sample in samples]
                    self.sample_ref_cache.extend(sample_refs)
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Get the current worker's node ID to pass to the DataCoordinator
                    # This is necessary because when the DataCoordinator receives a remote call,
                    # ray.get_runtime_context().get_node_id() returns the DataCoordinator's node_id,
                    # not the caller's node_id
                    caller_node_id = ray.get_runtime_context().get_node_id()
                    
                    put_future = self.data_coordinator.put_batch.remote(sample_infos, sample_refs, caller_node_id)
                    loop.run_until_complete(put_future)
                    logger.info(f"Rank {self._rank}: ✅ Successfully PUT {len(samples)} samples to DataCoordinator for key '{key}' (source_dp={source_dp_size}, dest_dp={dest_dp_size})")

        except Exception as e:
            logger.error(f"Rank {self._rank}: Unexpected error in put_data_to_buffers for key '{key}': {e}", exc_info=True)
            raise

    def get_data_from_buffers(
        self,
        key: str,
        cur_dp_size: int,
        cur_dp_rank: int,
        timing_raw: Dict[str, float]
    ) -> Optional[DataProto]:
        """
        Gets data from the DataCoordinator by filtering for a specific key,
        then collates the resulting Samples back into a single DataProto.
        
        Args:
            key: The key to filter samples
            cur_dp_size: Current node's DP size
            cur_dp_rank: Current worker's DP rank
            timing_raw: Timing dict for performance tracking
        """
        with timer(self.enable_perf, f"get_intern_data_{key}", timing_raw):
            if key in self.internal_data_cache:
                cached_data = self.internal_data_cache.pop(key)
                return cached_data
        def key_filter(sample_info: SampleInfo) -> bool:
            return sample_info.dict_info.get('key') == key

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        with timer(self.enable_perf, f"get_samples_from_coordinator_{key}", timing_raw):
            try:
                rollout_n = self.config.actor_rollout_ref.rollout.n if hasattr(self.config, 'actor_rollout_ref') else 1
            except (AttributeError, KeyError):
                rollout_n = 1
            
            if rollout_n is None or rollout_n < 1:
                rollout_n = 1
            
            adjusted_batch_size = int(self.config.data.train_batch_size * rollout_n / cur_dp_size)
            
            logger.info(
                f"Rank {self._rank}: Requesting from DataCoordinator: "
                f"key='{key}', cur_dp={cur_dp_size}, "
                f"adjusted_batch_size={adjusted_batch_size} (train_bs={self.config.data.train_batch_size} * rollout_n={rollout_n} / cur_dp={cur_dp_size})"
            )
            
            # Use filter_plugin to get only samples with matching key
            # Use balance_partitions to optimize sample distribution by length
            sample_refs = loop.run_until_complete(
                self.data_coordinator.get_batch.remote(
                    adjusted_batch_size,
                    cur_dp_rank,
                    filter_plugin=key_filter,
                    balance_partitions=cur_dp_size
                )
            )

        if not sample_refs:
            logger.warning(f"Rank {self._rank}: ❌ DataCoordinator returned EMPTY list for key '{key}' (adjusted_batch_size={adjusted_batch_size})")
            return None

        logger.info(f"Rank {self._rank}: ✅ Retrieved {len(sample_refs)} sample references from DataCoordinator for key '{key}'")

        with timer(self.enable_perf, f"ray_get_samples_{key}", timing_raw):
            samples = ray.get(sample_refs)

        with timer(self.enable_perf, f"collate_samples_{key}", timing_raw):
            # Collate the list of Sample objects back into a single DataProto
            tensordict = Samples2Dict(samples)

        return tensordict

    def reset_data_buffer(self):
        """
        DEPRECATED with DataCoordinator. The get calls are now consuming.
        This can be a no-op, but for safety, we could implement a clear if needed.
        For now, it does nothing as intended.
        """
        logger.debug("`reset_data_buffer` is a no-op with the new DataCoordinator model as gets are consuming.")
        if self._rank == 0:
            self.data_coordinator.reset_cache.remote()

    def _get_node_process_group(self, node: Node) -> ProcessGroup:
        """Retrieves the PyTorch ProcessGroup assigned to a specific graph node."""
        assignment = self.process_group_manager.get_node_assignment(node.node_id)
        if not (assignment and (name := assignment.get("process_group_name"))):
            raise ValueError(f"Process group assignment or name not found for node {node.node_id}.")

        pg = self.process_groups.get(name)
        if pg is None:
            raise ValueError(f"Process group '{name}' for node {node.node_id} was not created or found.")
        return pg

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

    def get_zeromq_address(self):
        return self.zmq_address

    def multi_agent_put_log(self, key: str, data: TensorDict, agent_group: int, next_dp_size: int, timing_raw):
        # This logic needs to be adapted to the new model. For now, it's a warning.
        logger.warning("`multi_agent_put_log` is not yet refactored for DataCoordinator and is a no-op.")
        pass

    def check_mode(self):
        return self.rollout_mode == 'sync' and self._multi_agent == False