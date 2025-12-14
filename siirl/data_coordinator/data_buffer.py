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

import asyncio
from typing import Dict, List, Optional, Tuple, Callable, Any
import heapq
import random
import ray
import loguru
import time
from collections import deque, defaultdict
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from siirl.data_coordinator.sample import SampleInfo
from siirl.utils.model_utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions


@ray.remote
class DataCoordinator:
    """
    A globally unique central Actor responsible for coordinating data producers (RolloutWorkers)
    and consumers (Trainers). It does not store the actual sample data, only the sample
    metadata (SampleInfo) and object references (ObjectRef). This allows it to implement
    complex global sampling strategies at a very low cost.
    """
    def __init__(self, nnodes:int, ppo_mini_batch_size: int, world_size: int):
        self.nnodes = nnodes
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.world_size = world_size
        # Use a deque to store tuples of metadata and references for efficient FIFO operations
        self._sample_queue: deque[Tuple[SampleInfo, ray.ObjectRef]] = deque()
        self._put_counter = 0  # Used for round-robin buffer selection
        self.lock = asyncio.Lock()
        loguru.logger.info("Global DataCoordinator initialized.")
        self._cache = []
        
    async def put(self, sample_info: SampleInfo, sample_ref: Any, caller_node_id: Optional[str] = None):
        """
        Called by a RolloutWorker to register a new sample reference and its metadata.
        This method automatically routes the ObjectRef to a DataBuffer on its local
        node to be held.
        
        Args:
            sample_info: Metadata about the sample
            sample_ref: Ray ObjectRef or the actual sample data
            caller_node_id: The node ID of the caller. If None, will try to get it from
                          the runtime context (but this won't work correctly for remote calls)
        """
        # Due to Ray's small object optimization, an ObjectRef passed by the client
        # might be automatically resolved to its actual value. Here, we ensure that
        # we are always handling an ObjectRef.
        if not isinstance(sample_ref, ray.ObjectRef):
            sample_ref = ray.put(sample_ref)

        # 1. Get the node ID of the caller
        # Note: When called remotely, ray.get_runtime_context().get_node_id() returns
        # the node ID of the DataCoordinator actor, not the caller. So we require the
        # caller to pass their node_id explicitly.
        if caller_node_id is None:
            caller_node_id = ray.get_runtime_context().get_node_id()
            loguru.logger.warning(
                "DataCoordinator.put() called without caller_node_id. "
                f"Using DataCoordinator's node_id {caller_node_id[:16]}... which may be incorrect."
            )

        # 2. Inject the node ID into SampleInfo for subsequent filtering
        #    Only inject if node_id has not been manually set, to facilitate testing.
        if sample_info.node_id is None:
            sample_info.node_id = caller_node_id

        # 4. Register the metadata and reference to the global queue
        async with self.lock:
            # More complex logic can be implemented here, such as inserting into a
            # priority queue based on priority
            self._sample_queue.append((sample_info, sample_ref))

    async def put_batch(self, sample_infos: List[SampleInfo], sample_refs: List[ray.ObjectRef], caller_node_id: Optional[str] = None):
        """
        Called by a worker to register a batch of new sample references and their metadata.
        This method routes the ObjectRefs to DataBuffers on their local nodes.
        
        Args:
            sample_infos: List of metadata for each sample
            sample_refs: List of Ray ObjectRefs
            caller_node_id: The node ID of the caller. If None, will try to get it from
                          the runtime context (but this won't work correctly for remote calls)
        """
        if not sample_refs:
            return

        # Get the node ID of the caller
        # Note: When called remotely, ray.get_runtime_context().get_node_id() returns
        # the node ID of the DataCoordinator actor, not the caller. So we require the
        # caller to pass their node_id explicitly.
        if caller_node_id is None:
            caller_node_id = ray.get_runtime_context().get_node_id()
            loguru.logger.warning(
                "DataCoordinator.put_batch() called without caller_node_id. "
                f"Using DataCoordinator's node_id {caller_node_id[:16]}... which may be incorrect."
            )

        for i in range(len(sample_infos)):
            if sample_infos[i].node_id is None:
                sample_infos[i].node_id = caller_node_id
        
        async with self.lock:
            self._sample_queue.extend(zip(sample_infos, sample_refs))

    async def get_batch(
        self, 
        batch_size: int, 
        dp_rank: int, 
        filter_plugin: Optional[Callable[[SampleInfo], bool]] = None,
        balance_partitions: Optional[int] = None
    ) -> List[ray.ObjectRef]:
        """Called by a Trainer to get a batch of sample ObjectRefs.
        
        Supports an optional filter plugin to implement custom sampling logic, and an
        optional length balancing feature.
        
        Args:
            batch_size: The requested batch size.
            filter_plugin: optional filters function for custom sampling logic.
            balance_partitions: If specified, the returned samples will be optimized
                              for even distribution among the given number of workers,
                              balancing the sum of sequence lengths for each worker.
                              Defaults to None (no length balancing).
        
        Returns:
            A list of sample ObjectRefs. If length balancing is enabled, the order
            of samples will be optimized.
        """
        async with self.lock:
            # No filter plugin, use efficient FIFO
            if len(self._cache) > 0:
                res = self._cache[dp_rank]
                return res
            if not filter_plugin:
                if len(self._sample_queue) < batch_size * balance_partitions:
                    loguru.logger.warning(f"Coordinator queue size ({len(self._sample_queue)}) is less than requested batch size ({batch_size}). Returning empty list.")
                    return []
        
                batch_items = []
                # Efficient O(batch_size) implementation using deque's O(1) popleft
                while self._sample_queue:
                    item = self._sample_queue.popleft()
                    batch_items.append(item)
                # Apply length balancing if requested
                if balance_partitions and balance_partitions > 1:
                    batch_refs = self._apply_length_balancing(batch_items, balance_partitions)
                else:
                    batch_refs = [item[1] for item in batch_items]
                self._cache = batch_refs
                get_refs = self._cache[:batch_size]
                self._cache = self._cache[batch_size:] if len(self._cache) >= batch_size else None
                return get_refs
            # With filter plugin, use O(N) filtering and reconstruction
            else:
                # 1. The filtering process does not consume elements from the queue
                if isinstance(filter_plugin, list):
                    potential_items = []
                    all_items =  [item for item in self._sample_queue ]
                    for item in self._sample_queue:
                        if all(filter_func(item[0]) for filter_func in filter_plugin):
                            potential_items.append(item)
                else:
                    potential_items = [item for item in self._sample_queue if filter_plugin(item[0])]
                # 2. Check if there are enough samples
                global_batch_size = batch_size * balance_partitions
                if len(potential_items) < global_batch_size:
                    loguru.logger.warning(f"After filtering, {filter_plugin} coordinator has {len(potential_items)} samples, which is less than requested batch size ({global_batch_size}). Returning empty list.")
                    return []
                potential_items = potential_items[:global_batch_size]
                # 4. Efficiently remove the selected items from the original queue
                # Use ObjectRef (guaranteed unique and hashable) to identify items for removal
                refs_to_remove = {item[1] for item in potential_items}
                self._sample_queue = deque(item for item in self._sample_queue if item[1] not in refs_to_remove)
                # Apply length balancing if requested
                if balance_partitions and balance_partitions > 1:
                    batch_refs = self._apply_length_balancing(potential_items, balance_partitions)
                else:
                    batch_refs = [item[1] for item in potential_items]
                for rank in range(balance_partitions):
                    self._cache.append(batch_refs[rank * batch_size: (rank + 1) * batch_size])
                res = self._cache[dp_rank]
                
                return res

    
    def _apply_length_balancing(
        self, 
        batch_items: List[Tuple[SampleInfo, ray.ObjectRef]], 
        k_partitions: int,
        keep_mini_batch = False
    ) -> List[ray.ObjectRef]:
        """Applies the length balancing algorithm to reorder samples.
        Uses the LPT (Longest Processing Time) algorithm to reorder samples so that
        if they are evenly distributed among k_partitions workers, the sum of
        sample lengths for each worker is as balanced as possible.
        
        Supports Group N: samples with the same uid will be assigned to the same partition,
        ensuring correct group-relative advantage computation for GRPO and similar algorithms.
        
        Args:
            batch_items: A list of (SampleInfo, ObjectRef) tuples.
            k_partitions: The number of partitions (typically the DP size).
            keep_mini_batch: Whether to keep mini-batch structure during balancing.
            
        Returns:
            A reordered list of ObjectRefs.
        """
        # ========== Step 1: Group samples by uid ==========
        uid_to_indices = defaultdict(list)
        for idx, (sample_info, _) in enumerate(batch_items):
            uid = sample_info.uid if sample_info.uid is not None else str(idx)
            uid_to_indices[uid].append(idx)
        
        # Check if grouping is needed (max_group_size > 1 means we have Group N)
        max_group_size = max(len(indices) for indices in uid_to_indices.values()) if uid_to_indices else 1
        
        if max_group_size == 1:
            # No grouping needed, use original single-sample balancing logic
            return self._apply_length_balancing_single_sample(batch_items, k_partitions, keep_mini_batch)
        
        # ========== Step 2: Calculate workload for each Group ==========
        group_list = list(uid_to_indices.keys())  # All unique uids
        group_workloads = []
        for uid in group_list:
            indices = uid_to_indices[uid]
            # Group workload = sum of all samples' sum_tokens in the group
            total_tokens = sum(batch_items[i][0].sum_tokens for i in indices)
            group_workloads.append(total_tokens)
        
        # ========== Step 3: Balance Groups across partitions ==========
        workload_lst = calculate_workload(group_workloads)
        
        # Check if number of groups is divisible by k_partitions
        num_groups = len(group_list)
        if num_groups < k_partitions:
            loguru.logger.warning(
                f"Number of groups ({num_groups}) is less than partitions ({k_partitions}). "
                f"Some partitions will be empty. Falling back to single-sample balancing."
            )
            return self._apply_length_balancing_single_sample(batch_items, k_partitions, keep_mini_batch)
        
        equal_size = (num_groups % k_partitions == 0)
        if not equal_size:
            loguru.logger.warning(
                f"Number of groups ({num_groups}) is not divisible by partitions ({k_partitions}). "
                f"Some partitions may have uneven group counts."
            )
        
        # Partition groups across workers
        group_partitions = get_seqlen_balanced_partitions(
            workload_lst, 
            k_partitions=k_partitions, 
            equal_size=equal_size
        )
        
        # ========== Step 4: Expand groups to samples, keeping group integrity ==========
        reordered_refs = []
        for partition_group_indices in group_partitions:
            for group_idx in partition_group_indices:
                uid = group_list[group_idx]
                sample_indices = uid_to_indices[uid]
                # Add all samples of the same group together, preserving original order within group
                for sample_idx in sample_indices:
                    reordered_refs.append(batch_items[sample_idx][1])
        
        loguru.logger.debug(
            f"Applied GROUP-aware length balancing: "
            f"{len(batch_items)} samples in {num_groups} groups (group_size={max_group_size}) "
            f"reordered into {k_partitions} partitions"
        )
        
        return reordered_refs

    def _apply_length_balancing_single_sample(
        self, 
        batch_items: List[Tuple[SampleInfo, ray.ObjectRef]], 
        k_partitions: int,
        keep_mini_batch = False
    ) -> List[ray.ObjectRef]:
        """
        This is used when there's no Group N (each uid has only one sample).
        
        Args:
            batch_items: A list of (SampleInfo, ObjectRef) tuples.
            k_partitions: The number of partitions (typically the DP size).
            keep_mini_batch: Whether to keep mini-batch structure during balancing.
            
        Returns:
            A reordered list of ObjectRefs.
        """
        # Extract the length of each sample.
        # Use sum_tokens as the length metric (includes prompt + response).
        seqlen_list = [item[0].sum_tokens for item in batch_items]
        
        # Use the karmarkar_karp balance
        workload_lst = calculate_workload(seqlen_list)
        # Decouple the DP balancing and mini-batching.
        if keep_mini_batch:
            minibatch_size = self.ppo_mini_batch_size
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(self.world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=self.world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=self.world_size, equal_size=True
            )    
            
            
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        
        # Reorder the samples based on the partitioning result.
        # Concatenate the partitions in order: [all samples from partition_0, all from partition_1, ...]
        reordered_refs = []
        for partition in global_partition_lst:
            for original_idx in partition:
                reordered_refs.append(batch_items[original_idx][1])
        
        loguru.logger.debug(
            f"Applied length balancing: {len(batch_items)} samples reordered into {k_partitions} partitions"
        )
        
        return reordered_refs
        

    async def get_all_by_filter(self, filter_plugin: Callable[[SampleInfo], bool]) -> List[ray.ObjectRef]:
        """
        Gets ALL sample ObjectRefs that match the filter plugin, consuming them from the queue.
        This is useful for pipeline-based data passing where a downstream stage needs the
        entire output of an upstream stage.
        """
        async with self.lock:
            # 1. Find all items that match the filter.
            items_to_return = [item for item in self._sample_queue if filter_plugin(item[0])]
            
            if not items_to_return:
                return []

            # 2. Extract their ObjectRefs.
            batch_refs = [item[1] for item in items_to_return]

            # 3. Efficiently remove the selected items from the original queue.
            refs_to_remove = {ref for ref in batch_refs}
            self._sample_queue = deque(item for item in self._sample_queue if item[1] not in refs_to_remove)
            
            return batch_refs

    async def get_valid_size(self) -> int:
        """Returns the number of samples in the current queue."""
        async with self.lock:
            return len(self._sample_queue)
    
    async def peek_source_dp_size(self, filter_plugin: Callable[[SampleInfo], bool]) -> Optional[int]:
        """
        Peek at the source_dp_size of matching samples without consuming them.
        
        Args:
            filter_plugin: Filter function to find matching samples
            
        Returns:
            The source_dp_size if found, None otherwise
        """
        async with self.lock:
            for sample_info, _ in self._sample_queue:
                if filter_plugin(sample_info):
                    source_dp_size = sample_info.dict_info.get('source_dp_size')
                    if source_dp_size is not None:
                        return source_dp_size
            return None

    def reset_cache(self):
        loguru.logger.warning("reset datacoordinator")
        self._sample_queue.clear()
        self._cache = []

    def __repr__(self) -> str:
        return f"<DataCoordinator(total_samples={len(self._sample_queue)})>"


# ====================================================================
# Initialization Logic
# ====================================================================

def init_data_coordinator(num_buffers: int, ppo_mini_batch_size: int, world_size: int) -> ray.actor.ActorHandle:
    """
    Initializes the data coordination system, which includes a global DataCoordinator
    and multiple distributed DataBuffers. Returns a single, unified DataCoordinator
    handle to the user.

    Args:
        num_buffers: The number of distributed DataBuffer instances to create,
                     usually equal to the number of nodes or total GPUs.
        force_local: If True, forces all Buffers to be created on the local node,
                     for single-machine testing.

    Returns:
        The Actor handle for the DataCoordinator.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before calling init_data_coordinator.")

    # 1. Create or get the globally unique DataCoordinator
    # Use a global name to ensure the coordinator's uniqueness
    coordinator_name = "global_data_coordinator"
    try:
        coordinator = ray.get_actor(coordinator_name)
        loguru.logger.info(f"Connected to existing DataCoordinator actor '{coordinator_name}'.")
    except ValueError:
        loguru.logger.info(f"Creating new DataCoordinator actor with global name '{coordinator_name}'.")
        coordinator = DataCoordinator.options(name=coordinator_name, lifetime="detached").remote(nnodes=num_buffers, ppo_mini_batch_size=ppo_mini_batch_size, world_size=world_size)
   
    return coordinator
