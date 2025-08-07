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
from siirl.utils.extras.device import get_device_id, get_device_name
from siirl.workers.dag.node import NodeRole
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

    def generate(self, worker_group_index: int, batch: DataProto, **kwargs) -> NodeOutput:
        """Generates sequences for a training batch using the rollout model."""
        if self.rollout_mode == "sync":
            gen_batch = self._prepare_generation_batch(batch)
            if self.config.actor_rollout_ref.rollout.name == "sglang":
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

    def compute_reward(self, batch: DataProto, tp_size: int, **kwargs) -> NodeOutput:
        """Calculates rewards for a batch of generated sequences."""
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

    def compute_ref_log_prob(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Computes log probabilities from the frozen reference model."""
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.REFERENCE].compute_ref_log_prob(batch)
        metrics = processed_data.meta_info.get("metrics", {})
        return NodeOutput(batch=processed_data, metrics=metrics)

    def compute_value(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Computes value estimates from the critic model."""
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.CRITIC].compute_values(batch)
        return NodeOutput(batch=processed_data)

    def compute_advantage(self, batch: DataProto, **kwargs) -> NodeOutput:
        """Computes advantages and returns for PPO using GAE."""
        adv_config = self.config.algorithm
        rollout_config = self.config.actor_rollout_ref.rollout
        if "token_level_rewards" not in batch.batch:
            # make sure rewards of angentB has been compute
            cur_node = kwargs["cur_node"]
            if depend_nodes := self.taskgraph.get_dependencies(cur_node.node_id):
                depend_node = depend_nodes[0]
                batch.batch["token_level_rewards"] = torch.zeros_like(batch.batch[f"agent_group_{depend_node.agent_group}_token_level_rewards"])
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

    def train_critic(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Performs a single training step on the critic model."""
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.CRITIC].update_critic(batch)
        process_group = self._get_node_process_group(self._get_node(NodeRole.CRITIC, worker_group_index))
        metrics = self._reduce_and_broadcast_metrics(processed_data.meta_info.get("metrics"), process_group)
        return NodeOutput(batch=processed_data, metrics=metrics)

    def train_actor(self, batch: DataProto, worker_group_index: int, **kwargs) -> NodeOutput:
        """Performs a single training step on the actor (policy) model."""
        if self.config.trainer.critic_warmup > self.global_steps:
            return NodeOutput(batch=batch)  # Skip actor update during critic warmup

        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        processed_data = self.agent_group_worker[worker_group_index][NodeRole.ACTOR].update_actor(batch)
        process_group = self._get_node_process_group(self._get_node(NodeRole.ACTOR, worker_group_index))
        metrics = self._reduce_and_broadcast_metrics(processed_data.meta_info.get("metrics"), process_group)
        return NodeOutput(batch=processed_data, metrics=metrics)

    def _get_rebalancing_context(self, current_node_id: str) -> Optional[Dict[str, Any]]:
        """
        Determines the distributed context for THIS node's rebalancing operation.
        It is self-contained and does not depend on downstream nodes.

        Args:
            current_node_id: The ID of the node currently executing.

        Returns:
            A dictionary with the distributed context if applicable, otherwise None.
        """
        current_node = self.taskgraph.get_node(current_node_id)
        try:
            my_process_group = self._get_node_process_group(current_node)
            dp_size, dp_rank, tp_rank, tp_size = self._get_node_dp_info(current_node)
        except (ValueError, AttributeError):
            # This can happen if the node is not part of a distributed setup, which is valid.
            return None

        return {"my_pg": my_process_group, "dp_size": dp_size, "my_dp_rank": dp_rank, "my_tp_rank": tp_rank, "tp_size": tp_size}

    def _gather_global_batch_counts(self, local_count: int, context: Dict[str, Any]) -> List[int]:
        """
        Gathers data counts from all TP Masters to form a global view,
        efficiently, without creating new process groups.
        """
        my_info = {"dp_rank": context["my_dp_rank"], "tp_rank": context["my_tp_rank"], "count": local_count}
        all_ranks_info = [None] * dist.get_world_size(group=context["my_pg"])
        dist.all_gather_object(all_ranks_info, my_info, group=context["my_pg"])

        master_infos = sorted([info for info in all_ranks_info if info["tp_rank"] == 0], key=lambda x: x["dp_rank"])
        return [info["count"] for info in master_infos]

    def _create_migration_plan(self, global_counts: List[int], target_batch_size: int, dp_world_size: int) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Creates a deterministic, globally consistent data migration plan."""
        # ## Step 1: Calculate the target number of prompts for each worker ##
        # Evenly distribute the target batch size, assigning any remainder to the first few workers.
        base_prompts_per_worker = target_batch_size // dp_world_size
        remainder = target_batch_size % dp_world_size
        target_counts = [base_prompts_per_worker + 1 if rank < remainder else base_prompts_per_worker for rank in range(dp_world_size)]

        # ## Step 2: Identify which workers have a data surplus and which have a deficit ##
        # A deficit means a worker has fewer prompts than its target.
        # A surplus means a worker has more prompts than its target.
        deficits = {rank: target_counts[rank] - count for rank, count in enumerate(global_counts) if count < target_counts[rank]}
        surpluses = {rank: count - target_counts[rank] for rank, count in enumerate(global_counts) if count > target_counts[rank]}

        # ## Step 3: Create the migration plan by matching surpluses to deficits ##
        # Sort by rank to ensure the matching process is deterministic across all workers.
        migration_plan = []
        deficit_workers = sorted(deficits.items())
        surplus_workers = sorted(surpluses.items())

        deficit_worker_index = 0
        surplus_worker_index = 0

        # Greedily match workers until either all deficits are filled or all surpluses are exhausted.
        while deficit_worker_index < len(deficit_workers) and surplus_worker_index < len(surplus_workers):
            # Get the current worker with a deficit (the "poor" worker)
            poor_worker_rank, needed = deficit_workers[deficit_worker_index]

            # Get the current worker with a surplus (the "rich" worker)
            rich_worker_rank, available = surplus_workers[surplus_worker_index]

            # Determine the amount of data to move in this step
            amount_to_move = min(needed, available)

            if amount_to_move > 0:
                migration_plan.append(
                    {
                        "from": rich_worker_rank,
                        "to": poor_worker_rank,
                        "amount": amount_to_move,
                    }
                )

            # Update the remaining amounts for the current workers
            deficit_workers[deficit_worker_index] = (poor_worker_rank, needed - amount_to_move)
            surplus_workers[surplus_worker_index] = (rich_worker_rank, available - amount_to_move)

            # If a worker's deficit is filled, move to the next one.
            if deficit_workers[deficit_worker_index][1] == 0:
                deficit_worker_index += 1

            # If a worker's surplus is exhausted, move to the next one.
            if surplus_workers[surplus_worker_index][1] == 0:
                surplus_worker_index += 1

        return migration_plan, target_counts

    def _execute_p2p_migration(self, plan: List[Dict[str, Any]], batch: DataProto, local_uids: List[str], context: Dict[str, Any]) -> List[DataProto]:
        """Executes data migration between TP masters using P2P send/recv with global ranks."""
        my_dp_rank = context["my_dp_rank"]
        tp_size = context["tp_size"]
        pg_ranks = dist.get_process_group_ranks(context["my_pg"])

        my_sends = [task for task in plan if task["from"] == my_dp_rank]
        my_receives = [task for task in plan if task["to"] == my_dp_rank]

        requests, received_buffers = [], {}
        device = get_device_name() if get_device_name() == "cpu" else f"{get_device_name()}:{get_device_id()}"

        for task in my_receives:
            sender_global_rank = pg_ranks[task["from"] * tp_size]
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
            dist.recv(tensor=size_tensor, src=sender_global_rank)
            if size_tensor.item() > 0:
                buffer = torch.empty(size_tensor.item(), dtype=torch.uint8, device=device)
                req = dist.irecv(tensor=buffer, src=sender_global_rank)
                requests.append(req)
                received_buffers[sender_global_rank] = buffer

        if my_sends:
            uids_to_send_pool = local_uids[-sum(task["amount"] for task in my_sends) :]
            for task in my_sends:
                dest_global_rank = pg_ranks[task["to"] * tp_size]
                uids, uids_to_send_pool = uids_to_send_pool[: task["amount"]], uids_to_send_pool[task["amount"] :]
                indices = [i for i, uid in enumerate(batch.non_tensor_batch["uid"]) if uid in uids]

                if not indices:
                    dist.send(tensor=torch.tensor([0], dtype=torch.long, device=device), dst=dest_global_rank)
                    continue

                serialized_data = pickle.dumps(batch[indices])
                dist.send(tensor=torch.tensor([len(serialized_data)], dtype=torch.long, device=device), dst=dest_global_rank)
                req = dist.isend(tensor=torch.from_numpy(np.frombuffer(serialized_data, dtype=np.uint8)).to(device), dst=dest_global_rank)
                requests.append(req)

        for req in requests:
            req.wait()

        return [pickle.loads(buf.cpu().numpy().tobytes()) for buf in received_buffers.values()]

    def _truncate_local_batch(self, batch: DataProto, target_count: int, local_uids: List[str]) -> DataProto:
        """Deterministically truncates the local batch to the target count."""
        if len(local_uids) <= target_count:
            return batch
        uids_to_keep = set(local_uids[:target_count])
        indices = [i for i, uid in enumerate(batch.non_tensor_batch["uid"]) if uid in uids_to_keep]
        return batch[indices]

    def _master_rebalance_logic(self, batch: DataProto, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encapsulates the entire rebalancing logic that is executed ONLY by the TP Master.

        Returns:
            The decision package to be synchronized with TP peers.
        """
        local_uids_count = len(set(batch.non_tensor_batch.get("uid", [])))
        global_counts = self._gather_global_batch_counts(local_uids_count, context)

        cache_was_used = hasattr(self, "sampling_leftover_cache") and self.sampling_leftover_cache and len(self.sampling_leftover_cache) > 0
        if cache_was_used:
            working_batch = DataProto.concat([self.sampling_leftover_cache, batch])
            self.sampling_leftover_cache = None
        else:
            working_batch = batch

        local_uids = sorted(list(set(working_batch.non_tensor_batch.get("uid", []))))

        if sum(global_counts) < self.config.data.train_batch_size:
            self.sampling_leftover_cache = working_batch
            return {"action": "cache", "cache_was_used": cache_was_used}
        else:
            plan, targets = self._create_migration_plan(global_counts, self.config.data.train_batch_size, context["dp_size"])
            received_shards = self._execute_p2p_migration(plan, working_batch, local_uids, dist.get_process_group_ranks(context["my_pg"]), context)

            my_target_count = targets[context["my_dp_rank"]]

            return {"action": "rebalance", "incremental_data": received_shards, "local_filter_info": {"target_count": my_target_count, "cache_was_used": cache_was_used}}

    def _synchronize_decision_to_peers(self, decision_package: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronizes the decision from the TP Master to its Peers using P2P communication.
        """
        sync_container = [decision_package]
        if context["tp_size"] > 1:
            pg_ranks = dist.get_process_group_ranks(context["my_pg"])
            dp_rank = context["my_dp_rank"]
            master_global_rank = pg_ranks[dp_rank * context["tp_size"]]

            if context["my_tp_rank"] == 0:
                for peer_tp_rank in range(1, context["tp_size"]):
                    peer_global_rank = pg_ranks[dp_rank * context["tp_size"] + peer_tp_rank]
                    dist.send_object(sync_container[0], dst=peer_global_rank)
            else:
                sync_container[0] = dist.recv_object(src=master_global_rank)

        return sync_container[0]

    def _reconstruct_batch_from_decision(self, batch: DataProto, decision: Dict[str, Any]) -> DataProto:
        """
        Reconstructs the final batch on ALL ranks based on the synchronized decision.
        """
        if decision["action"] == "cache":
            if decision["cache_was_used"]:
                working_batch = DataProto.concat([self.sampling_leftover_cache, batch])
            else:
                working_batch = batch
            self.sampling_leftover_cache = working_batch
            return DataProto()

        elif decision["action"] == "rebalance":
            filter_info = decision["local_filter_info"]

            if filter_info["cache_was_used"]:
                working_batch = DataProto.concat([self.sampling_leftover_cache, batch])
                self.sampling_leftover_cache = None
            else:
                working_batch = batch

            local_uids = sorted(list(set(working_batch.non_tensor_batch.get("uid", []))))

            local_filtered_batch = self._truncate_local_batch(working_batch, filter_info["target_count"], local_uids)

            incremental_data = decision["incremental_data"]
            if incremental_data:
                return DataProto.concat([local_filtered_batch] + incremental_data)
            else:
                return local_filtered_batch

        logger.error(f"Rank {self._rank}: Received unknown action '{decision.get('action')}' in sync package.")
        return batch  # Fallback to the original batch

    def postprocess_sampling(self, batch: DataProto, **kwargs) -> NodeOutput:
        """
        A stateful, distributed-aware node that rebalances data within its OWN process group.
        This is the final, robust, and highly optimized implementation with clear encapsulation.
        """
        context = self._get_rebalancing_context(kwargs["node_config"]["_node_id_"])

        if context is None:
            # This node is not part of a distributed setup, act as a passthrough.
            return NodeOutput(batch=batch)

        # Step 1: TP Master alone determines the rebalancing plan and decision.
        decision_package = None
        if context["my_tp_rank"] == 0:
            decision_package = self._master_rebalance_logic(batch, context)

        # Step 2: The decision is synchronized from the Master to all its Peers.
        final_decision = self._synchronize_decision_to_peers(decision_package, context)

        # Step 3: All ranks (Master and Peers) use the synchronized decision to build the final batch.
        final_batch = self._reconstruct_batch_from_decision(batch, final_decision)

        return NodeOutput(batch=final_batch)
