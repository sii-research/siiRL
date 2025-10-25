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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from siirl.utils.extras.device import get_device_id, get_device_name
from siirl.utils.params import SiiRLArguments
from siirl.workers.dag import TaskGraph
from siirl.workers.dag.node import NodeType
from siirl.workers.databuffer import DataProto


class DataRebalanceMixin:
    """
    A mixin providing robust, distributed-aware data rebalancing logic.

    This utility is designed for use after a distributed sampling phase where
    ranks may hold an uneven number of data samples. It makes a globally
    consistent decision to either rebalance the data to form a target batch
    size or cache it if the total is insufficient.
    """

    config: SiiRLArguments
    sampling_leftover_cache: Optional[DataProto]
    data_rebalance_masters_group: Optional[dist.ProcessGroup]
    taskgraph: Optional[TaskGraph]
    _rank: int

    _find_first_non_compute_ancestor: Any
    _get_node_process_group: Any
    _get_node_dp_info: Any
    _get_node_dp_info_dp_info: Any

    def _get_rebalancing_context(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        This method fetches the distributed context

        Args:
            node_id: The ID of the current node in the execution graph.

        Returns:
            A dictionary with the distributed context, or None if not applicable.
        """
        current_node = self.taskgraph.get_node(node_id)
        if not current_node:
            logger.debug(f"Rank {self._rank}: Could not find node '{node_id}'.")
            return None

        # If the node is a COMPUTE node, find its non-COMPUTE ancestor to get the
        # correct distributed context for the data's source.
        reference_node = current_node
        if current_node.node_type == NodeType.COMPUTE:
            ancestor = self._find_first_non_compute_ancestor(current_node.node_id)
            if ancestor:
                logger.debug(
                    f"Rank {self._rank}: Using context from ancestor '{ancestor.node_id}' for COMPUTE "
                    f"node '{current_node.node_id}'."
                )
                reference_node = ancestor
            else:
                logger.warning(
                    f"Rank {self._rank}: Could not find a non-COMPUTE ancestor for '{current_node.node_id}'."
                )
                return None

        try:
            pg = self._get_node_process_group(reference_node)
            dp_size, dp_rank, tp_rank, tp_size = self._get_node_dp_info(reference_node)
            device = (
                torch.device(f"{get_device_name()}:{get_device_id()}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        except (ValueError, AttributeError):
            # This is expected if the node is not part of a distributed setup.
            logger.debug(f"Rank {self._rank}: Node '{reference_node.node_id}' is not distributed.")
            return None

        return {
            "my_pg": pg,
            "dp_size": dp_size,
            "my_dp_rank": dp_rank,
            "my_tp_rank": tp_rank,
            "tp_size": tp_size,
            "device": device,
        }

    def _master_rebalance_logic(self, batch: DataProto, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core decision-making logic, executed only by master ranks (tp_rank=0).
        """
        if self.data_rebalance_masters_group is None:
            raise RuntimeError(
                "The 'data_rebalance_masters_group' has not been initialized. This "
                "dedicated group is required for safe rebalancing. Please "
                "initialize it and set it as an attribute on this instance."
            )

        local_new_count = len(set(batch.non_tensor_batch.get("uid", [])))

        cache_size = 0
        if self.sampling_leftover_cache and "uid" in self.sampling_leftover_cache.non_tensor_batch:
            cache_size = len(set(self.sampling_leftover_cache.non_tensor_batch["uid"]))

        # 1. Gather the state (local data counts) from all master ranks.
        my_state = {"dp_rank": context["my_dp_rank"], "count": local_new_count + cache_size}
        num_masters = context["dp_size"]
        global_state_list = [None] * num_masters
        dist.all_gather_object(global_state_list, my_state, group=self.data_rebalance_masters_group)
        global_state = sorted(global_state_list, key=lambda x: x["dp_rank"])

        # 2. Make a globally consistent decision based on the total data.
        global_counts = [s["count"] for s in global_state]
        total_prompts = sum(global_counts)
        target_batch_size = self.config.data.train_batch_size

        logger.debug(
            f"Rank {self._rank} (Master): Global counts: {global_counts}. Total: {total_prompts}. "
            f"Target: {target_batch_size}."
        )

        if total_prompts < target_batch_size:
            # Decision: Not enough data, cache everything for the next iteration.
            return {"action": "cache", "status": "INSUFFICIENT_DATA", "cache_was_used": cache_size > 0}
        else:
            # Decision: Data is sufficient, create a plan to rebalance.
            plan, targets = self._create_migration_plan(global_counts, target_batch_size, context["dp_size"])
            logger.debug(f"Rank {self._rank} (Master): Migration plan: {plan}. Targets: {targets}.")
            return {
                "action": "rebalance",
                "status": "OK",
                "plan": plan,
                "targets": targets,
                "cache_was_used": cache_size > 0,
            }

    def _synchronize_decision_to_peers(
        self, decision_package: Optional[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronizes the decision from the master rank to its TP peers.
        """
        if context["tp_size"] <= 1:
            return decision_package

        # The object must be in a list for the broadcast API.
        obj_list = [decision_package]

        # The source of the broadcast is the master rank (tp_rank=0) of this DP group.
        # We find its global rank to use as the `src`.
        master_global_rank = context["my_dp_rank"] * context["tp_size"]

        # Broadcast on the default WORLD group. The `src` argument ensures that
        # only the correct master sends the data.
        dist.broadcast_object_list(obj_list, src=master_global_rank, group=None)

        return obj_list[0]

    def _reconstruct_batch_from_decision(
        self, batch: DataProto, decision: Dict[str, Any], context: Dict[str, Any]
    ) -> DataProto:
        """
        Executes the synchronized decision on all ranks.

        This method is the single source of truth for updating the cache and
        constructing the final batch for the current step.
        """
        action = decision.get("action")

        if action == "cache":
            working_batch = batch
            if decision.get("cache_was_used") and self.sampling_leftover_cache:
                working_batch = DataProto.concat([self.sampling_leftover_cache, batch])

            self.sampling_leftover_cache = working_batch
            uid_count = len(set(working_batch.non_tensor_batch.get("uid", [])))
            logger.debug(f"Rank {self._rank}: Caching {uid_count} unique items. Returning empty batch.")
            return DataProto()

        elif action == "rebalance":
            working_batch = batch
            if decision.get("cache_was_used") and self.sampling_leftover_cache:
                working_batch = DataProto.concat([self.sampling_leftover_cache, batch])

            # The cache is consumed in a rebalance operation; clear it.
            self.sampling_leftover_cache = None
            received_shards = self._execute_p2p_migration(decision["plan"], working_batch, context)
            local_batch_to_keep = self._truncate_local_batch(working_batch, decision["plan"], context)
            final_batch = (
                DataProto.concat([local_batch_to_keep] + received_shards) if received_shards else local_batch_to_keep
            )

            my_target_count = decision["targets"][context["my_dp_rank"]]
            final_uid_count = len(set(final_batch.non_tensor_batch.get("uid", [])))
            summary_log_message = (
                f"Rank {self._rank}: Rebalance executed. Summary:\n"
                f"\t- Status        : {decision.get('status', 'UNKNOWN')} (Cache Used: "
                f"{decision.get('cache_was_used', False)})\n"
                f"\t- Data Movement : Received {len(received_shards)} shards\n"
                f"\t- Final Batch   : {final_uid_count} UIDs (Target: {my_target_count})"
            )
            logger.debug(summary_log_message)
            return final_batch

        logger.error(f"Rank {self._rank}: Received unknown action '{action}'. Passing batch through without changes.")
        self.sampling_leftover_cache = None
        return batch

    def _create_migration_plan(
        self, global_counts: List[int], target_batch_size: int, dp_world_size: int
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Creates a deterministic data migration plan based on global state."""
        base_items, remainder = divmod(target_batch_size, dp_world_size)
        target_counts = [base_items + 1 if rank < remainder else base_items for rank in range(dp_world_size)]

        deficits = {
            rank: target - count
            for rank, (count, target) in enumerate(zip(global_counts, target_counts))
            if count < target
        }
        surpluses = {
            rank: count - target
            for rank, (count, target) in enumerate(zip(global_counts, target_counts))
            if count > target
        }

        plan = []
        deficit_items, surplus_items = sorted(deficits.items()), sorted(surpluses.items())
        d_idx, s_idx = 0, 0
        while d_idx < len(deficit_items) and s_idx < len(surplus_items):
            poor_rank, needed = deficit_items[d_idx]
            rich_rank, available = surplus_items[s_idx]
            amount = min(needed, available)
            if amount > 0:
                plan.append({"from": rich_rank, "to": poor_rank, "amount": amount})
            deficit_items[d_idx] = (poor_rank, needed - amount)
            surplus_items[s_idx] = (rich_rank, available - amount)
            if deficit_items[d_idx][1] == 0:
                d_idx += 1
            if surplus_items[s_idx][1] == 0:
                s_idx += 1
        return plan, target_counts

    def _execute_p2p_migration(
        self, plan: List[Dict[str, Any]], batch: DataProto, context: Dict[str, Any]
    ) -> List[DataProto]:
        """Executes data migration between ranks using point-to-point communication."""
        my_dp_rank = context["my_dp_rank"]
        tp_size = context["tp_size"]
        device = context["device"]

        my_sends = [task for task in plan if task["from"] == my_dp_rank]
        my_receives = [task for task in plan if task["to"] == my_dp_rank]

        uids = sorted(list(set(batch.non_tensor_batch.get("uid", []))))
        uid_to_indices = {uid: [] for uid in uids}
        for i, uid in enumerate(batch.non_tensor_batch.get("uid", [])):
            uid_to_indices[uid].append(i)

        requests, received_buffers = [], {}

        # 1. Post all non-blocking receives.
        for task in my_receives:
            sender_global_rank = task["from"] * tp_size
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
            dist.recv(tensor=size_tensor, src=sender_global_rank)  # Blocking recv for size
            if size_tensor.item() > 0:
                buffer = torch.empty(size_tensor.item(), dtype=torch.uint8, device=device)
                req = dist.irecv(tensor=buffer, src=sender_global_rank)
                requests.append(req)
                received_buffers[sender_global_rank] = buffer

        # 2. Execute all sends.
        if my_sends:
            uids_to_send = uids[-sum(task["amount"] for task in my_sends) :]
            for task in my_sends:
                dest_global_rank = task["to"] * tp_size
                uids_for_this_send, uids_to_send = uids_to_send[: task["amount"]], uids_to_send[task["amount"] :]

                indices = [idx for uid in uids_for_this_send for idx in uid_to_indices[uid]]

                serialized = pickle.dumps(batch[indices])
                size_tensor = torch.tensor([len(serialized)], dtype=torch.long, device=device)
                dist.send(tensor=size_tensor, dst=dest_global_rank)  # Blocking send for size

                tensor_to_send = torch.from_numpy(np.frombuffer(serialized, dtype=np.uint8)).to(device)
                req = dist.isend(tensor=tensor_to_send, dst=dest_global_rank)
                requests.append(req)

        # 3. Wait for all transfers to complete.
        for req in requests:
            req.wait()

        return [pickle.loads(buf.cpu().numpy().tobytes()) for buf in received_buffers.values()]

    def _truncate_local_batch(self, batch: DataProto, plan: List[Dict[str, Any]], context: Dict[str, Any]) -> DataProto:
        """Deterministically truncates the local batch to the size it should be before receiving data."""
        num_to_send = sum(task["amount"] for task in plan if task["from"] == context["my_dp_rank"])

        uids = sorted(list(set(batch.non_tensor_batch.get("uid", []))))
        num_to_keep = len(uids) - num_to_send

        if num_to_keep <= 0:
            return DataProto()

        uids_to_keep = set(uids[:num_to_keep])
        indices = [i for i, uid in enumerate(batch.non_tensor_batch["uid"]) if uid in uids_to_keep]
        return batch[indices]

    def rebalance_sampled_data(self, batch: DataProto, node_id: str) -> DataProto:
        """
        Stateful, distributed-aware method to rebalance data across ranks.

        This is the main public entry point. It orchestrates the entire process
        of deciding whether to cache or rebalance data and executes that decision.

        Args:
            batch: The newly generated data on the current rank.
            node_id: The ID of the current node, used to fetch the correct
                     distributed context.

        Returns:
            The processed DataProto batch for the current training step. Returns
            an empty DataProto if the data was successfully cached.
        """
        logger.debug(f"Rank {self._rank}: Starting data rebalancing for node '{node_id}'.")
        context = self._get_rebalancing_context(node_id)
        if not context:
            logger.debug(f"Rank {self._rank}: Node is not distributed. Skipping rebalance.")
            return batch

        # The master rank (tp_rank=0) of each DP group makes the decision.
        decision_package = None
        if context["my_tp_rank"] == 0:
            decision_package = self._master_rebalance_logic(batch, context)

        # The decision is then synchronized to all other ranks in the TP group.
        final_decision = self._synchronize_decision_to_peers(decision_package, context)

        # All ranks execute the same decision to reconstruct the final batch.
        final_batch = self._reconstruct_batch_from_decision(batch, final_decision, context)

        status = final_decision.get("status", "UNKNOWN")
        final_uid_count = len(set(final_batch.non_tensor_batch.get("uid", [])))
        logger.debug(
            f"Rank {self._rank}: Finished rebalancing for node '{node_id}'. "
            f"Final unique UIDs: {final_uid_count}, Status: {status}."
        )
        return final_batch
