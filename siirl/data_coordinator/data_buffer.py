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
from collections import deque
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from siirl.data_coordinator.sample import SampleInfo




def get_seqlen_balanced_partitions_constrained_lpt(seqlen_list: List[int], k_partitions: int) -> List[List[int]]:
    """Partitions items into k subsets of equal item count with balanced sums.

    This function implements a constrained version of the LPT (Longest
    Processing Time) heuristic. It strictly adheres to the constraint that each
    partition must have a nearly equal number of items, and then uses the LPT
    principle to balance the sum of sequence lengths within that constraint.
    This is the recommended approach when a fixed number of items per worker
    is a hard requirement.

    Args:
        seqlen_list: A list of integers representing the "size" of each item.
        k_partitions: The desired number of partitions.

    Returns:
        A list of lists, where each inner list contains the original indices
        of the items assigned to that partition. Each list will have a size of
        len(seqlen_list) // k or len(seqlen_list) // k + 1.
    """
    if k_partitions <= 0:
        raise ValueError("Number of partitions (k_partitions) must be positive.")
    num_items = len(seqlen_list)
    
    # Ensure the data size is perfectly divisible.
    # If this fails, it indicates an unexpected issue in the data pipeline.
    if num_items % k_partitions != 0:
        loguru.logger.warning(
            f"Data size ({num_items}) is not evenly divisible by the number of partitions ({k_partitions}). "
            f"This may lead to uneven partition sizes."
        )
    
    # 1. Sort items by length in descending order, preserving original indices.
    indexed_lengths = sorted(enumerate(seqlen_list), key=lambda x: x[1], reverse=True)
    
    # 2. Determine the target number of items for each partition.
    base_size = num_items // k_partitions
    rem = num_items % k_partitions
    partition_target_sizes = [base_size + 1] * rem + [base_size] * (k_partitions - rem)
    
    # 3. Initialize partitions and a min-heap to track partition sums.
    #    The heap stores tuples of (current_sum, partition_index).
    partitions = [[] for _ in range(k_partitions)]
    partition_heap = [(0, i) for i in range(k_partitions)]
    heapq.heapify(partition_heap)
    
    # 4. Iterate through sorted items and assign each to a non-full partition
    #    with the smallest current sum.
    for original_idx, length in indexed_lengths:
        # Find the smallest, non-full partition.
        # Pop from the heap until we find a partition that is not yet full.
        while True:
            smallest_sum, smallest_idx = heapq.heappop(partition_heap)
            
            # Check if the selected partition is already full.
            if len(partitions[smallest_idx]) < partition_target_sizes[smallest_idx]:
                # This partition is not full, so we can assign the item.
                partitions[smallest_idx].append(original_idx)
                new_sum = smallest_sum + length
                
                # If the partition is still not full after adding, push it back.
                if len(partitions[smallest_idx]) < partition_target_sizes[smallest_idx]:
                    heapq.heappush(partition_heap, (new_sum, smallest_idx))
                break



def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions

def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """
    Calculates partitions of indices from seqlen_list such that the sum of sequence lengths
    in each partition is balanced. Uses the Karmarkar-Karp differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                           Requires len(seqlen_list) to be divisible by k_partitions.
                           If False, partitions can have varying numbers of items, focusing
                           only on balancing the sum of sequence lengths.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                         original indices of the items assigned to that partition. The indices
                         within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)

@ray.remote
class DataCoordinator:
    """
    A globally unique central Actor responsible for coordinating data producers (RolloutWorkers)
    and consumers (Trainers). It does not store the actual sample data, only the sample
    metadata (SampleInfo) and object references (ObjectRef). This allows it to implement
    complex global sampling strategies at a very low cost.
    """
    def __init__(self, nnodes:int):
        self.nnodes = nnodes
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
        k_partitions: int
    ) -> List[ray.ObjectRef]:
        """Applies the length balancing algorithm to reorder samples.
        
        Uses the LPT (Longest Processing Time) algorithm to reorder samples so that
        if they are evenly distributed among k_partitions workers, the sum of
        sample lengths for each worker is as balanced as possible.
        
        Args:
            batch_items: A list of (SampleInfo, ObjectRef) tuples.
            k_partitions: The number of partitions (typically the DP size).
            
        Returns:
            A reordered list of ObjectRefs.
        """
        # Extract the length of each sample.
        # Use sum_tokens as the length metric (includes prompt + response).
        seqlen_list = [item[0].sum_tokens for item in batch_items]
        
        try:
            # Use the LPT algorithm to calculate the optimal partitions.
            partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions, True)
            
            # Reorder the samples based on the partitioning result.
            # Concatenate the partitions in order: [all samples from partition_0, all from partition_1, ...]
            reordered_refs = []
            for partition in partitions:
                for original_idx in partition:
                    reordered_refs.append(batch_items[original_idx][1])
            
            loguru.logger.debug(
                f"Applied length balancing: {len(batch_items)} samples reordered into {k_partitions} partitions"
            )
            
            return reordered_refs
            
        except Exception as e:
            loguru.logger.warning(
                f"Failed to apply length balancing: {e}. Falling back to original order."
            )
            # If length balancing fails, return the original order.
            return [item[1] for item in batch_items]

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

def init_data_coordinator(num_buffers: int) -> ray.actor.ActorHandle:
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
        coordinator = DataCoordinator.options(name=coordinator_name, lifetime="detached").remote(nnodes=num_buffers)
   
    return coordinator
