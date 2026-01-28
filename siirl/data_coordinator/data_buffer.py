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
    def __init__(self, nnodes: int, ppo_mini_batch_size: int, world_size: int):
        self.nnodes = nnodes
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.world_size = world_size
        # Use a deque to store tuples of metadata and references for efficient FIFO operations
        self._sample_queue: deque[Tuple[SampleInfo, ray.ObjectRef]] = deque()
        self._put_counter = 0  # Used for round-robin buffer selection
        self.lock = asyncio.Lock()
        loguru.logger.info("Global DataCoordinator initialized.")
        
        # Cache for multi-rank access: [[rank0_data], [rank1_data], ...]
        self._cache: List[List[ray.ObjectRef]] = []
        self._cache_key: Optional[str] = None  # Track which key the cache belongs to
        
        # === Statistics tracking for dynamic sampling scenarios ===
        self._stats_batches_received = 0      # Number of put_batch calls
        self._stats_samples_received = 0      # Total samples received since last dispatch
        self._stats_accumulation_start = None # Time when accumulation started
        self._stats_last_progress_pct = 0     # Last logged progress percentage
        
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

        # 3. Register the metadata and reference to the global queue + update statistics
        async with self.lock:
            self._sample_queue.append((sample_info, sample_ref))
            # Update statistics (single sample put)
            self._stats_batches_received += 1
            self._stats_samples_received += 1
            if self._stats_accumulation_start is None:
                self._stats_accumulation_start = time.time()

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
            
            # Update statistics
            self._stats_batches_received += 1
            self._stats_samples_received += len(sample_refs)
            if self._stats_accumulation_start is None:
                self._stats_accumulation_start = time.time()

    async def get_batch(
        self, 
        batch_size: int, 
        dp_rank: int, 
        filter_plugin: Optional[Callable[[SampleInfo], bool]] = None,
        balance_partitions: Optional[int] = None,
        cache_key: Optional[str] = None
    ) -> List[ray.ObjectRef]:
        """Called by a Trainer to get a batch of sample ObjectRefs.
        
        Supports caching for multi-rank access within the same node, filter plugins
        for custom sampling logic, and length balancing across partitions.
        
        Args:
            batch_size: The requested batch size per partition.
            dp_rank: The data parallel rank requesting data.
            filter_plugin: Optional filter function for custom sampling logic.
            balance_partitions: Number of partitions for length balancing (typically dp_size).
            cache_key: Key to identify the cache (e.g., node name like 'compute_reward').
                       Different keys invalidate the cache to prevent data mixing.
        
        Returns:
            A list of sample ObjectRefs for the specified dp_rank.
        """
        async with self.lock:
            global_batch_size = batch_size * balance_partitions
            
            # === Phase 1: Check cache validity ===
            if self._cache:
                if cache_key is not None and self._cache_key == cache_key:
                    # Same key: return cached data (allows multiple reads by tp/pp ranks)
                    if dp_rank < len(self._cache):
                        return self._cache[dp_rank]
                    return []
                # Different key: invalidate old cache
                self._cache = []
                self._cache_key = None
            
            # === Phase 2: Fetch data from queue ===
            if filter_plugin:
                # With filter: O(N) scan
                if isinstance(filter_plugin, list):
                    batch_items = [item for item in self._sample_queue 
                                   if all(f(item[0]) for f in filter_plugin)]
                else:
                    batch_items = [item for item in self._sample_queue 
                                   if filter_plugin(item[0])]
                
                if len(batch_items) < global_batch_size:
                    self._log_accumulation_progress(len(batch_items), global_batch_size)
                    return []
                
                # Take only what we need and remove from queue
                batch_items = batch_items[:global_batch_size]
                refs_to_remove = {item[1] for item in batch_items}
                self._sample_queue = deque(
                    item for item in self._sample_queue if item[1] not in refs_to_remove
                )
            else:
                # No filter: efficient FIFO
                if len(self._sample_queue) < global_batch_size:
                    self._log_accumulation_progress(len(self._sample_queue), global_batch_size)
                    return []
                
                batch_items = [self._sample_queue.popleft() for _ in range(global_batch_size)]
            
            # === Phase 3: Apply length balancing ===
            if balance_partitions and balance_partitions > 1:
                batch_refs = self._apply_length_balancing(batch_items, balance_partitions)
            else:
                batch_refs = [item[1] for item in batch_items]
            
            # === Phase 4: Build cache (unified nested list structure) ===
            self._cache = [
                batch_refs[rank * batch_size:(rank + 1) * batch_size]
                for rank in range(balance_partitions)
            ]
            self._cache_key = cache_key
            
            self._log_dispatch_stats(global_batch_size)
            return self._cache[dp_rank]

    def _log_accumulation_progress(self, current_samples: int, target_samples: int):
        """Log progress milestones at INFO level when reaching 25%, 50%, 75%."""
        if target_samples <= 0:
            return
            
        current_pct = int(current_samples * 100 / target_samples)
        
        # Log at 25%, 50%, 75% milestones (only once per milestone)
        # Log the highest crossed milestone that hasn't been logged yet
        milestones = [25, 50, 75]
        highest_crossed = None
        for milestone in milestones:
            if current_pct >= milestone and self._stats_last_progress_pct < milestone:
                highest_crossed = milestone
        
        if highest_crossed is not None:
            wait_time = time.time() - self._stats_accumulation_start if self._stats_accumulation_start else 0
            loguru.logger.info(
                f"[DataCoordinator] Accumulation {highest_crossed}%: {current_samples}/{target_samples} samples "
                f"({self._stats_batches_received} batches, {wait_time:.1f}s)"
            )
            self._stats_last_progress_pct = highest_crossed
    
    def _log_dispatch_stats(self, dispatched_samples: int):
        """Log statistics when dispatching a batch and reset counters."""
        wait_time = time.time() - self._stats_accumulation_start if self._stats_accumulation_start else 0

        avg_samples_per_batch = (
            self._stats_samples_received / self._stats_batches_received
            if self._stats_batches_received > 0 else 0
        )

        total_received = self._stats_samples_received
        remaining_in_queue = total_received - dispatched_samples

        loguru.logger.info(
            f"[DataCoordinator DISPATCH] "
            f"Accumulated: {total_received} samples from {self._stats_batches_received} batches | "
            f"Dispatching: {dispatched_samples} samples | "
            f"Remaining in queue: {remaining_in_queue} | "
            f"Avg per batch: {avg_samples_per_batch:.1f} | "
            f"Wait: {wait_time:.1f}s"
        )

        self._stats_batches_received = 0
        self._stats_samples_received = 0
        self._stats_accumulation_start = None
        self._stats_last_progress_pct = 0
    
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

        equal_size = num_groups % k_partitions == 0
        if not equal_size:
            loguru.logger.warning(
                f"Number of groups ({num_groups}) is not divisible by partitions ({k_partitions}). "
                f"Some partitions may have uneven group counts."
            )

        # Partition groups across workers
        group_partitions = get_seqlen_balanced_partitions(workload_lst, k_partitions=k_partitions, equal_size=equal_size)

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
        keep_mini_batch=False,
    ) -> List[ray.ObjectRef]:
        """Original length balancing logic for single samples (no UID grouping).
        
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
        """Reset the coordinator state for a new training step."""
        loguru.logger.info("Resetting DataCoordinator cache")
        self._sample_queue.clear()
        self._cache = []
        self._cache_key = None

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
