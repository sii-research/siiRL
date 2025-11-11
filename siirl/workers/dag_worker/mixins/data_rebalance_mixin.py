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
            # Correctly unpack 6 values as specified by the user
            dp_size, dp_rank, tp_rank, tp_size, _, _ = self._get_node_dp_info(reference_node)
            device = (
                torch.device(f"{get_device_name()}:{get_device_id()}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        except (ValueError, AttributeError, TypeError) as e:
            # Catch TypeError in case _get_node_dp_info returns None
            logger.error(
                f"Rank {self._rank}: Failed to get distributed context for node '{reference_node.node_id}'. "
                f"This is expected if the node is not distributed, but could also be an error. Error: {e}"
            )
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
        This function is responsible *only* for deciding whether to cache or
        to rebalance, and then dispatching to the execution function.
        """
        if self.data_rebalance_masters_group is None:
            raise RuntimeError(
                "The 'data_rebalance_masters_group' has not been initialized. This "
                "dedicated group is required for safe rebalancing."
            )

        try:
            # Trust the master group as the single source of truth for DP size.
            num_masters = dist.get_world_size(self.data_rebalance_masters_group)
        except (RuntimeError, ValueError) as e:
            logger.error(f"Rank {self._rank}: Failed to get world size of 'data_rebalance_masters_group'. Error: {e}")
            return {"action": "error", "status": "ERROR_MASTER_GROUP"}

        local_new_count = len(set(batch.non_tensor_batch.get("uid", [])))

        cache_size = 0
        if self.sampling_leftover_cache and "uid" in self.sampling_leftover_cache.non_tensor_batch:
            cache_size = len(set(self.sampling_leftover_cache.non_tensor_batch["uid"]))

        # 1. Gather the state (local data counts) from all master ranks.
        my_state = {
            "dp_rank": context["my_dp_rank"], 
            "count": local_new_count + cache_size,
            "new_count": local_new_count,
            "cache_count": cache_size
        }
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

        # Print detailed data statistics in table format (only on global rank 0 to avoid duplicate logs)
        if self._rank == 0:
            total_new = sum(s["new_count"] for s in global_state)
            total_cached = sum(s["cache_count"] for s in global_state)
            table_rows = "\n".join([
                f"  {s['dp_rank']:<6} {s['new_count']:<15} {s['cache_count']:<10} {s['count']:<10}"
                for s in global_state
            ])
            logger.info(
                f"Rank {self._rank} (Master): Post-Sampling Data Statistics:\n"
                f"  {'Rank':<6} {'New Samples':<15} {'Cached':<10} {'Total':<10}\n"
                f"  {'-' * 45}\n"
                f"{table_rows}\n"
                f"  {'-' * 45}\n"
                f"  {'Total':<6} {total_new:<15} {total_cached:<10} {total_prompts:<10}\n"
                f"  Target Batch Size: {target_batch_size}\n"
                f"  Gap: {target_batch_size - total_prompts if total_prompts < target_batch_size else 0}"
            )

        cache_was_used = cache_size > 0
        if total_prompts < target_batch_size:
            # Decision: Not enough data, cache everything for the next iteration.
            decision = {"action": "cache", "status": "INSUFFICIENT_DATA", "cache_was_used": cache_was_used}
            logger.info(f"Rank {self._rank} (Master): Data insufficient. Decision: Cache.")
            # Print detailed shortage information (only on global rank 0)
            if self._rank == 0:
                shortage = target_batch_size - total_prompts
                logger.info(
                    f"Rank {self._rank} (Master): ⚠️  INSUFFICIENT DATA - Need to continue rollout sampling\n"
                    f"  Current Total: {total_prompts} UIDs\n"
                    f"  Target Batch:  {target_batch_size} UIDs\n"
                    f"  Shortage:      {shortage} UIDs ({shortage * 100.0 / target_batch_size:.1f}%)\n"
                    f"  Decision:      Cache current data and continue sampling"
                )
            return decision
        else:
            # Decision: Data is sufficient. Create plan and delegate to executor.
            # Print detailed surplus information (only on global rank 0)
            if self._rank == 0:
                surplus = total_prompts - target_batch_size
                logger.info(
                    f"Rank {self._rank} (Master): ✅ SUFFICIENT DATA - Proceeding to rebalance and train\n"
                    f"  Current Total: {total_prompts} UIDs\n"
                    f"  Target Batch:  {target_batch_size} UIDs\n"
                    f"  Surplus:       {surplus} UIDs"
                )
            plan, targets = self._create_migration_plan(global_counts, target_batch_size, num_masters)
            return self._execute_master_rebalance(batch, cache_was_used, plan, targets, context)

    def _execute_master_rebalance(
        self,
        batch: DataProto,
        cache_was_used: bool,
        plan: List[Dict[str, Any]],
        targets: List[int],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes the rebalancing plan (P2P migration) for a master rank."""
        logger.debug(f"Rank {self._rank} (Master): Data sufficient. Proceeding to rebalance.")
        logger.debug(f"Rank {self._rank} (Master): Migration plan: {plan}. Targets: {targets}.")

        # Create the full working batch (local + cache)
        working_batch = batch
        if cache_was_used and self.sampling_leftover_cache:
            working_batch = DataProto.concat([self.sampling_leftover_cache, batch])

        # Build UID artifacts *once* for performance
        uids, uid_to_indices_map = self._build_uid_artifacts(working_batch)

        # Master rank executes the P2P migration with other masters.
        received_shards = self._execute_p2p_migration(plan, working_batch, uids, uid_to_indices_map, context)

        my_target_count = targets[context["my_dp_rank"]]

        # The decision package now contains the *result* of the migration.
        decision = {
            "action": "rebalance",
            "status": "OK",
            "incremental_data": received_shards,
            "local_filter_info": {
                "target_count": my_target_count,
                "cache_was_used": cache_was_used,
            },
        }

        decision_summary = {
            "action": decision.get("action"),
            "status": decision.get("status"),
            "target_count": decision.get("local_filter_info", {}).get("target_count"),
            "cache_was_used": decision.get("local_filter_info", {}).get("cache_was_used"),
            "received_shards": len(decision.get("incremental_data", [])),
        }
        logger.debug(f"Rank {self._rank} (Master): Rebalance logic complete. Decision summary: {decision_summary}")
        return decision

    def _synchronize_decision_to_peers(
        self, decision_package: Optional[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronizes the decision from the master rank (tp_rank=0) to its TP peers
        using a robust two-phase P2P (send/recv) protocol.
        """
        tp_size = context.get("tp_size", 1)
        if tp_size <= 1:
            return decision_package

        device = context["device"]
        my_pg = context["my_pg"]
        tp_rank = context["my_tp_rank"]
        dp_rank = context["my_dp_rank"]

        try:
            pg_ranks = dist.get_process_group_ranks(my_pg)
        except (ValueError, AttributeError) as e:
            logger.error(
                f"Rank {self._rank}: Failed to get ranks from process group {my_pg}. P2P sync will fail. Error: {e}"
            )
            return {"action": "error", "status": "ERROR_RANK_RESOLUTION"}

        if tp_rank == 0:
            # --- MASTER (tp_rank=0) SENDS ---
            if decision_package is None:
                logger.warning(f"Rank {self._rank} (Master): Decision package is None, creating default error package.")
                decision_package = {"action": "error", "status": "ERROR_NO_DECISION"}

            try:
                serialized_decision = pickle.dumps(decision_package)
            except Exception as e:
                logger.error(f"Rank {self._rank} (Master): Failed to pickle decision package. Error: {e}")
                # Send 0 size as an error signal
                serialized_decision = bytearray()

            data_tensor = torch.from_numpy(np.frombuffer(serialized_decision, dtype=np.uint8)).to(device)
            size_tensor = torch.tensor([data_tensor.numel()], dtype=torch.long, device=device)

            requests = []
            for peer_tp_rank in range(1, tp_size):
                try:
                    peer_global_rank = pg_ranks[dp_rank * tp_size + peer_tp_rank]
                except IndexError:
                    logger.error(
                        f"Rank {self._rank}: Failed to find rank for DP={dp_rank}, TP={peer_tp_rank} in pg_ranks."
                    )
                    continue

                # Phase 1: Blocking send for size.
                dist.send(tensor=size_tensor, dst=peer_global_rank)

                # Phase 2: Non-blocking send for data (if size > 0).
                if size_tensor.item() > 0:
                    req = dist.isend(tensor=data_tensor, dst=peer_global_rank)
                    requests.append(req)

            # Wait for all async sends to complete
            for req in requests:
                req.wait()

            return decision_package

        else:
            # --- PEER (tp_rank > 0) RECEIVES ---
            try:
                master_global_rank = pg_ranks[dp_rank * tp_size]
            except IndexError:
                logger.error(f"Rank {self._rank}: Failed to find master rank for DP={dp_rank}, TP=0 in pg_ranks.")
                return {"action": "error", "status": "ERROR_RANK_RESOLUTION"}

            # Phase 1: Blocking receive for size.
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
            dist.recv(tensor=size_tensor, src=master_global_rank)

            buffer_size = size_tensor.item()
            if buffer_size == 0:
                logger.warning(f"Rank {self._rank}: Received 0-size decision package (or error signal from master).")
                return {"action": "error", "status": "ERROR_ZERO_SIZE_PACKAGE"}

            # Phase 2: Non-blocking receive for data.
            buffer = torch.empty(buffer_size, dtype=torch.uint8, device=device)
            request = dist.irecv(tensor=buffer, src=master_global_rank)
            request.wait()

            # Phase 3: Deserialize object.
            try:
                received_decision = pickle.loads(buffer.cpu().numpy().tobytes())
                return received_decision
            except Exception as e:
                logger.error(f"Rank {self._rank}: Failed to deserialize decision package. Error: {e}")
                return {"action": "error", "status": f"ERROR_DESERIALIZATION: {e}"}

    def _reconstruct_batch_from_decision(self, batch: DataProto, decision: Dict[str, Any]) -> DataProto:
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
            uid_count = 0
            if "uid" in working_batch.non_tensor_batch:
                uid_count = len(set(working_batch.non_tensor_batch.get("uid", [])))
            logger.info(f"Rank {self._rank}: Caching {uid_count} unique items. Returning empty batch.")
            return DataProto()

        elif action == "rebalance":
            local_filter_info = decision.get("local_filter_info", {})
            cache_was_used = local_filter_info.get("cache_was_used", False)

            working_batch = batch
            if cache_was_used and self.sampling_leftover_cache:
                working_batch = DataProto.concat([self.sampling_leftover_cache, batch])

            # The cache is consumed in a rebalance operation; clear it.
            self.sampling_leftover_cache = None

            # Get sorted UIDs and the index map *once* for performance.
            local_uids, uid_to_indices_map = self._build_uid_artifacts(working_batch)

            # Truncate the local (or local + cached) batch to its target size
            my_target_count = local_filter_info.get("target_count", 0)
            local_batch_to_keep = self._truncate_local_batch(
                working_batch, my_target_count, local_uids, uid_to_indices_map
            )

            # Get the shards that were received by the master and synced to this peer
            received_shards = decision.get("incremental_data", [])

            final_batch = (
                DataProto.concat([local_batch_to_keep] + received_shards) if received_shards else local_batch_to_keep
            )

            final_uid_count = 0
            if "uid" in final_batch.non_tensor_batch:
                final_uid_count = len(set(final_batch.non_tensor_batch.get("uid", [])))

            summary_log_message = (
                f"Rank {self._rank}: Rebalance executed. Summary:\n"
                f"\t- Status        : {decision.get('status', 'UNKNOWN')} (Cache Used: {cache_was_used})\n"
                f"\t- Data Movement : Received {len(received_shards)} shards (from master's sync)\n"
                f"\t- Final Batch   : {final_uid_count} UIDs (Target: {my_target_count})"
            )
            logger.info(summary_log_message)
            return final_batch

        logger.error(
            f"Rank {self._rank}: Received unknown or error action '{action}'. Passing batch through without changes."
        )
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
        self,
        plan: List[Dict[str, Any]],
        batch: DataProto,
        local_uids: List[str],
        uid_to_indices_map: Dict[str, List[int]],
        context: Dict[str, Any],
    ) -> List[DataProto]:
        """Executes data migration between master ranks using point-to-point communication."""
        my_dp_rank = context["my_dp_rank"]
        tp_size = context["tp_size"]
        device = context["device"]
        my_pg = context["my_pg"]

        try:
            # Resolve global ranks from the node's process group
            pg_ranks = dist.get_process_group_ranks(my_pg)
        except (ValueError, AttributeError) as e:
            logger.error(
                f"Rank {self._rank}: Failed to get ranks from process group {my_pg}. P2P will fail. Error: {e}"
            )
            return []

        my_sends = [task for task in plan if task["from"] == my_dp_rank]
        my_receives = [task for task in plan if task["to"] == my_dp_rank]

        requests, received_buffers = [], {}
        zero_size_tensor = torch.tensor([0], dtype=torch.long, device=device)

        # 1. Post all non-blocking receives.
        for task in my_receives:
            try:
                # We only communicate with other masters (tp_rank=0)
                sender_global_rank = pg_ranks[task["from"] * tp_size + 0]
            except IndexError:
                logger.error(f"Rank {self._rank}: Recv plan error. Cannot find rank for DP={task['from']} in pg_ranks.")
                continue

            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
            dist.recv(tensor=size_tensor, src=sender_global_rank)  # Blocking recv for size

            buffer_size = size_tensor.item()
            if buffer_size > 0:
                buffer = torch.empty(buffer_size, dtype=torch.uint8, device=device)
                req = dist.irecv(tensor=buffer, src=sender_global_rank)
                requests.append(req)
                received_buffers[sender_global_rank] = buffer
            else:
                logger.debug(f"Rank {self._rank}: Receiving 0 bytes from rank {sender_global_rank}.")

        # 2. Execute all sends.
        if my_sends:
            # Get UIDs to send from the *end* of the sorted list
            uids_to_send = local_uids[-sum(task["amount"] for task in my_sends) :]
            for task in my_sends:
                try:
                    # We only communicate with other masters (tp_rank=0)
                    dest_global_rank = pg_ranks[task["to"] * tp_size + 0]
                except IndexError:
                    logger.error(
                        f"Rank {self._rank}: Send plan error. Cannot find rank for DP={task['to']} in pg_ranks."
                    )
                    continue

                amount_to_send = task["amount"]
                if amount_to_send == 0:
                    dist.send(tensor=zero_size_tensor, dst=dest_global_rank)
                    continue

                uids_for_this_send, uids_to_send = uids_to_send[:amount_to_send], uids_to_send[amount_to_send:]

                # Use helper for efficient slicing
                indices = self._get_indices_from_uid_map(uid_to_indices_map, set(uids_for_this_send))

                if not indices:
                    logger.warning(f"Rank {self._rank}: Found no indices for UIDs to send to {dest_global_rank}.")
                    dist.send(tensor=zero_size_tensor, dst=dest_global_rank)
                    continue

                try:
                    serialized = pickle.dumps(batch[indices])
                except Exception as e:
                    logger.error(
                        f"Rank {self._rank}: Failed to pickle data for rank {dest_global_rank}. "
                        f"Sending 0 size. Error: {e}"
                    )
                    dist.send(tensor=zero_size_tensor, dst=dest_global_rank)
                    continue

                size_tensor = torch.tensor([len(serialized)], dtype=torch.long, device=device)
                dist.send(tensor=size_tensor, dst=dest_global_rank)  # Blocking send for size

                tensor_to_send = torch.from_numpy(np.frombuffer(serialized, dtype=np.uint8)).to(device)
                req = dist.isend(tensor=tensor_to_send, dst=dest_global_rank)
                requests.append(req)

        # 3. Wait for all transfers to complete.
        for req in requests:
            req.wait()

        # 4. Deserialize received data
        received_shards = []
        for rank, buf in received_buffers.items():
            try:
                shard = pickle.loads(buf.cpu().numpy().tobytes())
                received_shards.append(shard)
            except Exception as e:
                logger.error(
                    f"Rank {self._rank}: Failed to deserialize P2P data from rank {rank}. Skipping shard. Error: {e}"
                )

        return received_shards

    def _truncate_local_batch(
        self,
        batch: DataProto,
        target_count: int,
        local_uids: List[str],
        uid_to_indices_map: Dict[str, List[int]],
    ) -> DataProto:
        """Deterministically truncates the local batch to the target count."""

        num_to_keep = target_count

        if num_to_keep <= 0:
            logger.debug(f"Rank {self._rank}: Truncating local batch to 0.")
            return DataProto()

        if len(local_uids) <= num_to_keep:
            # We are keeping all (or more than) what we have, so just return the batch.
            return batch

        # Keep the *first* 'num_to_keep' UIDs from the sorted list
        uids_to_keep = set(local_uids[:num_to_keep])

        # Use helper for efficient slicing
        indices = self._get_indices_from_uid_map(uid_to_indices_map, uids_to_keep)

        logger.debug(f"Rank {self._rank}: Truncating local batch from {len(local_uids)} UIDs to {num_to_keep} UIDs.")
        return batch[indices]

    def _build_uid_artifacts(self, batch: DataProto) -> Tuple[List[str], Dict[str, List[int]]]:
        """
        Builds the sorted UID list and the UID-to-indices map in one pass.
        This is a CPU-intensive operation, so we centralize it.
        """
        uids_in_batch_list = batch.non_tensor_batch.get("uid", [])

        # Ensure it's a list, as it might be a numpy array
        if not isinstance(uids_in_batch_list, list):
            uids_in_batch_list = uids_in_batch_list.tolist()

        if not uids_in_batch_list:
            return [], {}

        uid_to_indices_map = {}
        unique_uids = set()

        for i, uid in enumerate(uids_in_batch_list):
            if uid not in uid_to_indices_map:
                uid_to_indices_map[uid] = []
                unique_uids.add(uid)
            uid_to_indices_map[uid].append(i)

        sorted_uids = sorted(list(unique_uids))
        return sorted_uids, uid_to_indices_map

    def _get_indices_from_uid_map(self, uid_to_indices_map: Dict[str, List[int]], uids_to_find: set) -> List[int]:
        """Efficiently retrieves all row indices corresponding to a set of UIDs."""
        all_indices = []
        for uid in uids_to_find:
            if uid in uid_to_indices_map:
                all_indices.extend(uid_to_indices_map[uid])
        return all_indices

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
            # Handle cache even for non-distributed nodes.
            if self.sampling_leftover_cache:
                logger.debug(f"Rank {self._rank}: Non-distributed node has cache. Concatenating.")
                final_batch = DataProto.concat([self.sampling_leftover_cache, batch])
                self.sampling_leftover_cache = None  # Consume cache
                return final_batch
            return batch

        # The master rank (tp_rank=0) of each DP group makes the decision.
        decision_package = None
        if context["my_tp_rank"] == 0:
            decision_package = self._master_rebalance_logic(batch, context)

        # The decision is then synchronized to all other ranks in the TP group.
        final_decision = self._synchronize_decision_to_peers(decision_package, context)

        # All ranks execute the same decision to reconstruct the final batch.
        final_batch = self._reconstruct_batch_from_decision(batch, final_decision)

        status = final_decision.get("status", "UNKNOWN")
        final_uid_count = 0
        if "uid" in final_batch.non_tensor_batch:
            final_uid_count = len(set(final_batch.non_tensor_batch.get("uid", [])))

        logger.debug(
            f"Rank {self._rank}: Finished rebalancing for node '{node_id}'. "
            f"Final unique UIDs: {final_uid_count}, Status: {status}."
        )

        # Add global consistency check, controlled by a config flag.
        if (
            context["my_tp_rank"] == 0
            and status == "OK"
            and getattr(self.config.data, "rebalance_consistency_check", False)
        ):
            try:
                target_batch_size = self.config.data.train_batch_size
                num_masters = dist.get_world_size(self.data_rebalance_masters_group)
                global_counts_list = [None] * num_masters
                dist.all_gather_object(global_counts_list, final_uid_count, group=self.data_rebalance_masters_group)

                total_final_count = sum(global_counts_list)
                if total_final_count != target_batch_size:
                    logger.warning(
                        f"Rank {self._rank} (Master): Rebalance consistency check FAILED. "
                        f"Final global count ({total_final_count}) does not match "
                        f"target batch size ({target_batch_size}). "
                        f"Master counts: {global_counts_list}"
                    )
                else:
                    logger.debug(
                        f"Rank {self._rank} (Master): Rebalance consistency check PASSED. "
                        f"Final global count: {total_final_count}"
                    )
            except Exception as e:
                logger.warning(f"Rank {self._rank} (Master): Failed to run consistency check. Error: {e}")

        return final_batch
