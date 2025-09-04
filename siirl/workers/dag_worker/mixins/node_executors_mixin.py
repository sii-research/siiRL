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

import pickle
import uuid
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import distributed as dist

from siirl.scheduler.reward import compute_reward
from siirl.utils.debug import DistProfiler
from siirl.utils.extras.device import get_device_id, get_device_name
from siirl.workers.dag.node import NodeRole, NodeType
from siirl.workers.dag_worker.algorithms import apply_kl_penalty, compute_advantage, compute_response_mask
from siirl.workers.dag_worker.core_algos import agg_loss
from siirl.workers.dag_worker.data_structures import NodeOutput
from siirl.workers.databuffer import DataProto


class NodeExecutorsMixin:
    """Contains the specific execution methods for different node roles in the DAG."""

    from typing import Any, Dict

    from siirl.utils.params import SiiRLArguments
    from siirl.workers.base_worker import Worker
    from siirl.workers.dag.node import NodeRole

    agent_group_worker: Dict[int, Dict[NodeRole, Worker]]
    config: SiiRLArguments
    reward_fn: Any
    kl_ctrl_in_reward: Any
    _rank: int
    global_steps: int

    _prepare_generation_batch: Any
    _get_node_process_group: Any
    _get_node: Any
    _reduce_and_broadcast_metrics: Any
    _find_first_non_compute_ancestor: Any

    @DistProfiler.annotate(role="generate")
    def generate(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Generates sequences for a training batch using the rollout model."""
        if self._multi_agent is False:
            if self.rollout_mode == 'sync':
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
            elif self._async_rollout_manager is not None:
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
        else:
            gen_batch = self._prepare_generation_batch(batch)
            if self.config.actor_rollout_ref.rollout.agent.rewards_with_env and "reward_model" in batch.non_tensor_batch:
                gen_batch.non_tensor_batch["reward_model"] = batch.non_tensor_batch["reward_model"] 
            assert self.config.actor_rollout_ref.rollout.name == 'sglang'
            gen_output = self.multi_agent_loop.generate_sequence(gen_batch)
            if gen_output:
                metrics = gen_output.meta_info.get("metrics", {})
                gen_output.meta_info = {}
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])
                batch = batch.repeat(self.config.actor_rollout_ref.rollout.n, interleave=True).union(gen_output)
                if "response_mask" not in batch.batch:
                    batch.batch["response_mask"] = compute_response_mask(batch)
                return NodeOutput(batch=batch, metrics=metrics) 
            return NodeOutput(batch=batch, metrics={})    
               
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
    def compute_advantage(self, batch: DataProto, **kwargs) -> NodeOutput:
        """Computes advantages and returns for PPO using GAE."""
        adv_config = self.config.algorithm
        rollout_config = self.config.actor_rollout_ref.rollout
        if "token_level_rewards" not in batch.batch:
            # make sure rewards of angentB has been compute
            cur_node = kwargs["cur_node"]
            if depend_nodes := self.taskgraph.get_dependencies(cur_node.node_id):
                depend_node = depend_nodes[0]
                if adv_config.share_reward_in_agent:
                    batch.batch["token_level_rewards"] = batch.batch[f"agent_group_{depend_node.agent_group}_token_level_rewards"].clone()
                else:    
                    batch.batch["token_level_rewards"] = torch.zeros_like(batch.batch[f"agent_group_{depend_node.agent_group}_token_level_rewards"])
                batch.batch["token_level_scores"] = batch.batch[f"agent_group_{depend_node.agent_group}_token_level_scores"].clone()
                node_output = self.compute_value(batch, cur_node.agent_group)
                node_output.batch.batch["pre_values"] = batch.batch[f"agent_group_{depend_node.agent_group}_values"]
                node_output.batch.batch["pre_advantages"] = batch.batch[f"agent_group_{depend_node.agent_group}_advantages"]

                batch = node_output.batch
            else:
                raise RuntimeError(f"cur_node {cur_node.node_id} have no rewards with can't find it's dependencies reward")
        return NodeOutput(
            batch=compute_advantage(
                batch, adv_estimator=adv_config.adv_estimator, gamma=adv_config.gamma, lam=adv_config.lam, num_repeat=rollout_config.n, norm_adv_by_std_in_grpo=adv_config.norm_adv_by_std_in_grpo, weight_factor_in_cpgd=adv_config.weight_factor_in_cpgd, multi_turn=rollout_config.multi_turn.enable
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

    def init_postsampling_process_group(world_size: int, tp_size: int) -> Optional[dist.ProcessGroup]:
        """
        Initializes a custom process group containing only the master ranks (tp_rank=0).

        This function should be called once during startup, after the default
        process group has been initialized. It is used to create a safe communication
        channel for collective operations among a subset of ranks.

        Args:
            world_size (int): The total number of processes in the default world.
            tp_size (int): The size of the tensor parallel group.

        Returns:
            A new ProcessGroup object for the master ranks, or None if no group
            is needed (i.e., tp_size <= 1 or world_size <= 1).
        """
        # The group is only necessary for distributed training with tensor parallelism.
        if world_size <= 1 or tp_size <= 1:
            return None

        all_ranks = list(range(world_size))

        # Identify master ranks (those with tp_rank == 0).
        # In a typical setup, rank 'r' has tp_rank = r % tp_size.
        master_ranks = [rank for rank in all_ranks if (rank % tp_size) == 0]

        # Create the dedicated process group for these master ranks.
        postsampling_masters_group = dist.new_group(ranks=master_ranks)

        return postsampling_masters_group

    def _get_rebalancing_context(self, current_node_id: str) -> Optional[Dict[str, Any]]:
        """
        Determines the distributed context for this node's rebalancing operation.
        If the node is of type COMPUTE, it intelligently finds its first non-COMPUTE
        ancestor and uses its distributed context, ensuring correct behavior.
        """
        current_node = self.taskgraph.get_node(current_node_id)
        if not current_node:
            logger.debug(f"Rank {self._rank}: Could not find node '{current_node_id}' in the task graph.")
            return None

        reference_node = current_node
        if current_node.node_type == NodeType.COMPUTE:
            ancestor = self._find_first_non_compute_ancestor(current_node.node_id)
            if ancestor:
                logger.debug(f"Rank {self._rank}: Node '{current_node.node_id}' is a COMPUTE node. Using context from its ancestor '{ancestor.node_id}'.")
                reference_node = ancestor
            else:
                logger.debug(f"Rank {self._rank}: Could not find a non-COMPUTE ancestor for COMPUTE node '{current_node.node_id}'. Cannot determine distributed context.")
                return None

        try:
            my_process_group = self._get_node_process_group(reference_node)
            dp_size, dp_rank, tp_rank, tp_size = self._get_node_dp_info(reference_node)
        except (ValueError, AttributeError) as e:
            logger.error(f"Rank {self._rank}: Could not get distributed context for reference node '{reference_node.node_id}'. This is expected if the node is not part of a distributed setup. Error: {e}")
            return None

        return {"my_pg": my_process_group, "dp_size": dp_size, "my_dp_rank": dp_rank, "my_tp_rank": tp_rank, "tp_size": tp_size}

    def _gather_global_state(self, local_new_count: int, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gathers both new and cached data counts from all TP Masters to form a
        complete and consistent global state view.
        """
        cache_size = 0
        if self.sampling_leftover_cache and "uid" in self.sampling_leftover_cache.non_tensor_batch:
            # BUG FIX: Calculate cache size based on unique UIDs, not total rows.
            cache_size = len(set(self.sampling_leftover_cache.non_tensor_batch["uid"]))

        my_state = {
            "dp_rank": context["my_dp_rank"],
            "tp_rank": context["my_tp_rank"],
            "new_count": local_new_count,
            "cached_count": cache_size,
        }
        all_ranks_state = [None] * dist.get_world_size(group=context["my_pg"])
        dist.all_gather_object(all_ranks_state, my_state, group=context["my_pg"])

        return sorted([state for state in all_ranks_state if state["tp_rank"] == 0], key=lambda x: x["dp_rank"])

    def _create_migration_plan(self, global_state: List[Dict[str, Any]], target_batch_size: int, dp_world_size: int) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Creates a deterministic data migration plan based on the complete global state,
        considering the total data (new + cached) on each rank.
        """
        current_counts = [state["new_count"] + state["cached_count"] for state in global_state]

        base_items = target_batch_size // dp_world_size
        remainder = target_batch_size % dp_world_size
        target_counts = [base_items + 1 if rank < remainder else base_items for rank in range(dp_world_size)]

        deficits = {rank: target_counts[rank] - count for rank, count in enumerate(current_counts) if count < target_counts[rank]}
        surpluses = {rank: count - target_counts[rank] for rank, count in enumerate(current_counts) if count > target_counts[rank]}

        migration_plan = []
        deficit_workers = sorted(deficits.items())
        surplus_workers = sorted(surpluses.items())

        d_idx, s_idx = 0, 0
        while d_idx < len(deficit_workers) and s_idx < len(surplus_workers):
            poor_rank, needed = deficit_workers[d_idx]
            rich_rank, available = surplus_workers[s_idx]
            amount = min(needed, available)
            if amount > 0:
                migration_plan.append({"from": rich_rank, "to": poor_rank, "amount": amount})

            deficit_workers[d_idx] = (poor_rank, needed - amount)
            surplus_workers[s_idx] = (rich_rank, available - amount)

            if deficit_workers[d_idx][1] == 0:
                d_idx += 1
            if surplus_workers[s_idx][1] == 0:
                s_idx += 1

        return migration_plan, target_counts

    def _execute_p2p_migration(self, plan: List[Dict[str, Any]], batch: DataProto, local_uids: List[str], context: Dict[str, Any]) -> List[DataProto]:
        """Executes data migration between ranks using P2P send/recv."""
        my_dp_rank, tp_size = context["my_dp_rank"], context["tp_size"]
        pg_ranks = dist.get_process_group_ranks(context["my_pg"])
        my_sends = [task for task in plan if task["from"] == my_dp_rank]
        my_receives = [task for task in plan if task["to"] == my_dp_rank]

        requests, received_buffers = [], {}
        device = get_device_name() if get_device_name() == "cpu" else f"{get_device_name()}:{get_device_id()}"

        for task in my_receives:
            sender_rank = pg_ranks[task["from"] * tp_size]
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
            dist.recv(tensor=size_tensor, src=sender_rank)
            if size_tensor.item() > 0:
                buffer = torch.empty(size_tensor.item(), dtype=torch.uint8, device=device)
                req = dist.irecv(tensor=buffer, src=sender_rank)
                requests.append(req)
                received_buffers[sender_rank] = buffer

        if my_sends:
            uids_to_send = local_uids[-sum(task["amount"] for task in my_sends) :]
            for task in my_sends:
                dest_rank = pg_ranks[task["to"] * tp_size]
                uids, uids_to_send = uids_to_send[: task["amount"]], uids_to_send[task["amount"] :]
                indices = [i for i, uid in enumerate(batch.non_tensor_batch["uid"]) if uid in uids]

                if not indices:
                    dist.send(tensor=torch.tensor([0], dtype=torch.long, device=device), dst=dest_rank)
                    continue

                serialized = pickle.dumps(batch[indices])
                dist.send(tensor=torch.tensor([len(serialized)], dtype=torch.long, device=device), dst=dest_rank)
                req = dist.isend(tensor=torch.from_numpy(np.frombuffer(serialized, dtype=np.uint8)).to(device), dst=dest_rank)
                requests.append(req)

        for req in requests:
            req.wait()

        return [pickle.loads(buf.cpu().numpy().tobytes()) for buf in received_buffers.values()]

    def _truncate_local_batch(self, batch: DataProto, target_count: int, local_uids: List[str]) -> DataProto:
        """Deterministically truncates the local batch to its target size."""
        if len(local_uids) <= target_count:
            return batch
        uids_to_keep = set(local_uids[:target_count])
        indices = [i for i, uid in enumerate(batch.non_tensor_batch["uid"]) if uid in uids_to_keep]
        return batch[indices]

    def _master_rebalance_logic(self, batch: DataProto, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        The core decision-making logic, executed by all master ranks (tp_rank=0).
        It now makes a consistent decision based on the true global state.
        """
        local_new_count = len(set(batch.non_tensor_batch.get("uid", [])))

        if self.postsampling_masters_group is None:
            raise RuntimeError("The dedicated 'postsampling_masters_group' has not been initialized. This group is required for rebalancing logic to prevent deadlocks. Please call 'init_postsampling_process_group' after 'dist.init_process_group' and set the returned group on this class instance.")

        # 1. Prepare the local state object for this master rank.
        cache_size = 0
        if self.sampling_leftover_cache and "uid" in self.sampling_leftover_cache.non_tensor_batch:
            cache_size = len(set(self.sampling_leftover_cache.non_tensor_batch["uid"]))

        my_state = {
            # The local rank within the masters_group corresponds to the DP rank.
            "dp_rank": dist.get_rank(group=self.postsampling_masters_group),
            "new_count": local_new_count,
            "cached_count": cache_size,
        }

        # 2. Perform the all-gather operation on the dedicated masters group.
        num_masters = dist.get_world_size(group=self.postsampling_masters_group)
        global_state_list = [None] * num_masters
        dist.all_gather_object(global_state_list, my_state, group=self.postsampling_masters_group)
        global_state = sorted(global_state_list, key=lambda x: x["dp_rank"])

        logger.debug(f"Rank {self._rank} (Master): Gathered global state: {global_state}.")

        total_new = sum(s["new_count"] for s in global_state)
        total_cached = sum(s["cached_count"] for s in global_state)
        total_prompts = total_new + total_cached
        cache_was_used = total_cached > 0

        target_batch_size = self.config.data.train_batch_size
        logger.debug(f"Rank {self._rank} (Master): Total prompts (new={total_new}, cached={total_cached}, total={total_prompts}). Target: {target_batch_size}.")

        if total_prompts < target_batch_size:
            decision = {"action": "cache", "status": "INSUFFICIENT_DATA", "cache_was_used": cache_was_used}
            logger.debug(f"Rank {self._rank} (Master): Data insufficient. Decision: {decision}")
            return decision
        else:
            logger.debug(f"Rank {self._rank} (Master): Data sufficient. Proceeding to rebalance.")
            working_batch = batch
            if self.sampling_leftover_cache:
                working_batch = DataProto.concat([self.sampling_leftover_cache, batch])

            plan, targets = self._create_migration_plan(global_state, target_batch_size, context["dp_size"])
            logger.debug(f"Rank {self._rank} (Master): Migration plan: {plan}. Targets: {targets}.")

            # The P2P migration still correctly uses the original `context` to resolve global ranks.
            received_shards = self._execute_p2p_migration(plan, working_batch, sorted(list(set(working_batch.non_tensor_batch.get("uid", [])))), context)
            my_target_count = targets[context["my_dp_rank"]]
            decision = {"action": "rebalance", "status": "OK", "incremental_data": received_shards, "local_filter_info": {"target_count": my_target_count, "cache_was_used": cache_was_used}}

            # Create a concise summary of the decision for logging.
            decision_summary = {
                "action": decision.get("action"),
                "status": decision.get("status"),
                "target_count": decision.get("local_filter_info", {}).get("target_count"),
                "cache_was_used": decision.get("local_filter_info", {}).get("cache_was_used"),
                "received_shards": len(decision.get("incremental_data", [])),
            }
            logger.debug(f"Rank {self._rank} (Master): Rebalance complete. Decision summary: {decision_summary}")
            return decision

    def _synchronize_decision_to_peers(self, decision_package: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronizes a decision package from the master rank (tp_rank=0) to its peers
        within the same tensor-parallel group.

        This function uses a robust, two-phase asynchronous point-to-point (P2P) protocol:
        1.  A synchronous exchange of the data size to ensure receivers can allocate buffers correctly.
        2.  An asynchronous transfer of the actual data payload to maximize network parallelism.

        Args:
            decision_package: The Python dictionary to be synchronized. This is only provided on the
                            master rank (tp_rank=0).
            context: A dictionary containing the distributed context, including tp_rank, tp_size, etc.

        Returns:
            The synchronized decision package, now available on all ranks in the TP group.
        """
        tp_size = context.get("tp_size", 1)

        # --- Base Case: No synchronization needed if there's only one rank in the group. ---
        if tp_size <= 1:
            return decision_package

        # --- Setup common distributed variables ---
        pg_ranks = dist.get_process_group_ranks(context["my_pg"])
        dp_rank = context["my_dp_rank"]
        tp_rank = context["my_tp_rank"]
        device = get_device_name() if get_device_name() == "cpu" else f"{get_device_name()}:{get_device_id()}"

        if tp_rank == 0:
            # Ensure we have a valid object to send.
            if decision_package is None:
                logger.warning("Decision package is None on master, creating a default error package.")
                decision_package = {"action": "error", "status": "ERROR_NO_DECISION"}

            # 1. Serialize the Python object into a byte buffer.
            serialized_decision = pickle.dumps(decision_package)
            data_tensor = torch.from_numpy(np.frombuffer(serialized_decision, dtype=np.uint8)).to(device)
            size_tensor = torch.tensor([data_tensor.numel()], dtype=torch.long, device=device)

            # --- Phase 1: Synchronously send the size to all peers. ---
            # This blocking step is a crucial synchronization point. It guarantees that
            # all receivers are ready and know the data size before we start sending the payload.
            for peer_tp_rank in range(1, tp_size):
                peer_global_rank = pg_ranks[dp_rank * tp_size + peer_tp_rank]
                dist.send(tensor=size_tensor, dst=peer_global_rank)

            # --- Phase 2: Asynchronously send the data payload to all peers. ---
            # This allows all send operations to be in-flight simultaneously.
            requests = []
            for peer_tp_rank in range(1, tp_size):
                peer_global_rank = pg_ranks[dp_rank * tp_size + peer_tp_rank]
                req = dist.isend(tensor=data_tensor, dst=peer_global_rank)
                requests.append(req)

            # Wait for all asynchronous send operations to complete.
            for req in requests:
                req.wait()

            return decision_package

        else:
            master_global_rank = pg_ranks[dp_rank * tp_size]

            # --- Phase 1: Synchronously receive the size. ---
            # This blocking receive ensures we have the correct size before allocating a buffer.
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
            dist.recv(tensor=size_tensor, src=master_global_rank)

            # --- Phase 2: Asynchronously receive the data payload. ---
            # Allocate the buffer and post the non-blocking receive operation.
            buffer_size = size_tensor.item()
            buffer = torch.empty(buffer_size, dtype=torch.uint8, device=device)
            request = dist.irecv(tensor=buffer, src=master_global_rank)

            # Wait for the data to be fully received.
            request.wait()

            # Deserialize the byte buffer back into a Python object.
            received_decision = pickle.loads(buffer.cpu().numpy().tobytes())

            return received_decision

    def _reconstruct_batch_from_decision(self, batch: DataProto, decision: Dict[str, Any]) -> DataProto:
        """
        Reconstructs the final data batch on all ranks based on the synchronized decision.
        This method is the single source of truth for updating `self.sampling_leftover_cache`.
        """
        action = decision.get("action")
        if action == "cache":
            if decision.get("cache_was_used") and self.sampling_leftover_cache:
                working_batch = DataProto.concat([self.sampling_leftover_cache, batch])
            else:
                working_batch = batch
            self.sampling_leftover_cache = working_batch
            if "uid" in self.sampling_leftover_cache.non_tensor_batch:
                logger.debug(f"Rank {self._rank}: Cache updated. New unique cache size: {len(set(self.sampling_leftover_cache.non_tensor_batch['uid']))}. Returning empty DataProto.")
            return DataProto()
        elif action == "rebalance":
            working_batch = batch
            if decision["local_filter_info"]["cache_was_used"] and self.sampling_leftover_cache:
                working_batch = DataProto.concat([self.sampling_leftover_cache, batch])

            self.sampling_leftover_cache = None

            local_uids = sorted(list(set(working_batch.non_tensor_batch.get("uid", []))))
            target_count = decision["local_filter_info"]["target_count"]
            local_filtered_batch = self._truncate_local_batch(working_batch, target_count, local_uids)
            incremental_data = decision["incremental_data"]

            final_batch = DataProto.concat([local_filtered_batch] + incremental_data) if incremental_data else local_filtered_batch
            final_uid_count = len(set(final_batch.non_tensor_batch.get("uid", [])))
            logger.debug(f"Rank {self._rank}: Rebalance complete. Final batch unique UIDs: {final_uid_count}.")
            return final_batch

        logger.error(f"Rank {self._rank}: Received unknown or error action '{action}' in sync package.")
        self.sampling_leftover_cache = None
        return batch

    def postprocess_sampling(self, batch: DataProto, **kwargs) -> NodeOutput:
        """
        A stateful, distributed-aware node that rebalances data across ranks.
        It makes a globally consistent decision on whether to cache or rebalance.
        """
        node_id = kwargs.get("node_config", {}).get("_node_id_", "Unknown")
        logger.debug(f"Rank {self._rank}: --- Starting postprocess_sampling for node '{node_id}' ---")
        context = self._get_rebalancing_context(node_id)
        if context is None:
            logger.debug(f"Rank {self._rank}: Node is not distributed. Passing data through.")
            return NodeOutput(batch=batch, metrics={"postprocess_status": "OK_SINGLE_NODE"})

        decision_package = None
        if context["my_tp_rank"] == 0:
            decision_package = self._master_rebalance_logic(batch, context)

        final_decision = self._synchronize_decision_to_peers(decision_package, context)
        final_batch = self._reconstruct_batch_from_decision(batch, final_decision)
        status = final_decision.get("status", "UNKNOWN")

        final_uid_count = len(set(final_batch.non_tensor_batch.get("uid", []))) if final_batch else 0
        logger.debug(f"Rank {self._rank}: --- Finished postprocess_sampling for node '{node_id}'. Final unique UIDs: {final_uid_count}, Status: {status} ---")
        return NodeOutput(batch=final_batch, metrics={"postprocess_status": status})
