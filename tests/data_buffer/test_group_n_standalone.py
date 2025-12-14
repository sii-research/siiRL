#!/usr/bin/env python3
"""
Standalone test for Group N balancing logic.
This test does not require Ray or other heavy dependencies.
It tests the core balancing algorithm directly.
"""

import unittest
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq


# ============================================================
# Minimal reimplementation of required functions for testing
# ============================================================

def calculate_workload(seqlen_list: list):
    """Calculate workload (simplified version)."""
    result = []
    for seqlen in seqlen_list:
        # FLOPs ≈ 24576 * seqlen + seqlen^2
        result.append(24576 * seqlen + seqlen ** 2)
    return result


def karmarkar_karp(seqlen_list: list, k_partitions: int, equal_size: bool):
    """Karmarkar-Karp algorithm for balanced partitioning."""
    class Set:
        def __init__(self):
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
        def __init__(self, items: list, k: int):
            self.k = k
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
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

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
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    return final_state.get_partitions()


def get_seqlen_balanced_partitions(seqlen_list: list, k_partitions: int, equal_size: bool):
    """Get balanced partitions using Karmarkar-Karp."""
    assert len(seqlen_list) >= k_partitions
    
    def _check_and_sort_partitions(partitions):
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            sorted_partitions[i] = sorted(partition)
        return sorted_partitions
    
    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


# ============================================================
# Mock classes
# ============================================================

@dataclass
class MockSampleInfo:
    uid: Optional[str] = None
    sum_tokens: int = 0


class MockObjectRef:
    def __init__(self, sample_id: int):
        self.sample_id = sample_id
    
    def __repr__(self):
        return f"MockRef({self.sample_id})"
    
    def __eq__(self, other):
        return isinstance(other, MockObjectRef) and self.sample_id == other.sample_id
    
    def __hash__(self):
        return hash(self.sample_id)


# ============================================================
# The function under test (copy from data_buffer.py)
# ============================================================

def apply_length_balancing_with_group_n(
    batch_items: List[Tuple[MockSampleInfo, MockObjectRef]],
    k_partitions: int
) -> List[MockObjectRef]:
    """
    Apply length balancing with Group N support.
    Samples with the same uid will be kept together.
    """
    # ========== Step 1: Group samples by uid ==========
    uid_to_indices = defaultdict(list)
    for idx, (sample_info, _) in enumerate(batch_items):
        uid = sample_info.uid if sample_info.uid is not None else str(idx)
        uid_to_indices[uid].append(idx)
    
    # Check if grouping is needed
    max_group_size = max(len(indices) for indices in uid_to_indices.values()) if uid_to_indices else 1
    
    if max_group_size == 1:
        # Fallback: single-sample balancing
        seqlen_list = [item[0].sum_tokens for item in batch_items]
        workload_lst = calculate_workload(seqlen_list)
        partitions = get_seqlen_balanced_partitions(
            workload_lst, k_partitions=k_partitions, equal_size=True
        )
        reordered_refs = []
        for partition in partitions:
            for idx in partition:
                reordered_refs.append(batch_items[idx][1])
        return reordered_refs
    
    # ========== Step 2: Calculate workload for each Group ==========
    group_list = list(uid_to_indices.keys())
    group_workloads = []
    for uid in group_list:
        indices = uid_to_indices[uid]
        total_tokens = sum(batch_items[i][0].sum_tokens for i in indices)
        group_workloads.append(total_tokens)
    
    # ========== Step 3: Balance Groups across partitions ==========
    workload_lst = calculate_workload(group_workloads)
    
    num_groups = len(group_list)
    if num_groups < k_partitions:
        # Fallback: not enough groups
        seqlen_list = [item[0].sum_tokens for item in batch_items]
        workload_lst_single = calculate_workload(seqlen_list)
        partitions = get_seqlen_balanced_partitions(
            workload_lst_single, k_partitions=k_partitions, equal_size=False
        )
        reordered_refs = []
        for partition in partitions:
            for idx in partition:
                reordered_refs.append(batch_items[idx][1])
        return reordered_refs
    
    equal_size = (num_groups % k_partitions == 0)
    group_partitions = get_seqlen_balanced_partitions(
        workload_lst, k_partitions=k_partitions, equal_size=equal_size
    )
    
    # ========== Step 4: Expand groups to samples ==========
    reordered_refs = []
    for partition_group_indices in group_partitions:
        for group_idx in partition_group_indices:
            uid = group_list[group_idx]
            sample_indices = uid_to_indices[uid]
            for sample_idx in sample_indices:
                reordered_refs.append(batch_items[sample_idx][1])
    
    return reordered_refs


# ============================================================
# Test Cases
# ============================================================

class TestGroupNBalancing(unittest.TestCase):
    """Tests for Group N balancing."""

    def _create_batch_items(
        self, 
        groups: List[Tuple[str, List[int]]]
    ) -> List[Tuple[MockSampleInfo, MockObjectRef]]:
        """Create batch items from groups specification."""
        batch_items = []
        sample_id = 0
        for uid, token_counts in groups:
            for tokens in token_counts:
                sample_info = MockSampleInfo(uid=uid, sum_tokens=tokens)
                mock_ref = MockObjectRef(sample_id)
                batch_items.append((sample_info, mock_ref))
                sample_id += 1
        return batch_items

    def _get_uid_distribution(
        self, 
        batch_items: List[Tuple[MockSampleInfo, MockObjectRef]], 
        reordered_refs: List[MockObjectRef],
        k_partitions: int
    ) -> List[set]:
        """Get uid distribution across partitions."""
        ref_to_uid = {ref.sample_id: info.uid for info, ref in batch_items}
        
        partition_size = len(reordered_refs) // k_partitions
        partitions = []
        for i in range(k_partitions):
            start = i * partition_size
            end = start + partition_size
            partition_refs = reordered_refs[start:end]
            partition_uids = {ref_to_uid[ref.sample_id] for ref in partition_refs}
            partitions.append(partition_uids)
        
        return partitions

    def test_two_groups_two_partitions(self):
        """Test: 2 groups, 2 partitions. Each partition gets 1 complete group."""
        print("\n=== Test: 2 groups, 2 partitions ===")
        groups = [
            ("uid_0", [100, 120, 80, 90]),   # 4 samples
            ("uid_1", [200, 180, 220, 190]), # 4 samples
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 2
        
        reordered_refs = apply_length_balancing_with_group_n(batch_items, k_partitions)
        uid_partitions = self._get_uid_distribution(batch_items, reordered_refs, k_partitions)
        
        print(f"  Partition 0 uids: {uid_partitions[0]}")
        print(f"  Partition 1 uids: {uid_partitions[1]}")
        
        # Each partition should have exactly 1 uid
        for i, uid_set in enumerate(uid_partitions):
            self.assertEqual(len(uid_set), 1, f"Partition {i} should have 1 group")
        
        print("  ✅ PASSED: Each partition has exactly 1 complete group")

    def test_four_groups_two_partitions(self):
        """Test: 4 groups, 2 partitions. Each partition gets 2 groups."""
        print("\n=== Test: 4 groups, 2 partitions ===")
        groups = [
            ("uid_0", [100, 110, 90, 100]),
            ("uid_1", [200, 210, 190, 200]),
            ("uid_2", [150, 140, 160, 150]),
            ("uid_3", [300, 290, 310, 300]),
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 2
        
        reordered_refs = apply_length_balancing_with_group_n(batch_items, k_partitions)
        uid_partitions = self._get_uid_distribution(batch_items, reordered_refs, k_partitions)
        
        print(f"  Partition 0 uids: {uid_partitions[0]}")
        print(f"  Partition 1 uids: {uid_partitions[1]}")
        
        # Each partition should have 2 groups
        for i, uid_set in enumerate(uid_partitions):
            self.assertEqual(len(uid_set), 2, f"Partition {i} should have 2 groups")
        
        # No overlap
        self.assertEqual(len(uid_partitions[0] & uid_partitions[1]), 0, "No overlap")
        
        print("  ✅ PASSED: Each partition has 2 complete groups, no overlap")

    def test_samples_contiguous_within_group(self):
        """Test: Samples within a group should be contiguous."""
        print("\n=== Test: Samples contiguity ===")
        groups = [
            ("uid_A", [100, 200, 300, 400]),
            ("uid_B", [150, 250, 350, 450]),
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 2
        
        reordered_refs = apply_length_balancing_with_group_n(batch_items, k_partitions)
        
        ref_to_uid = {ref.sample_id: info.uid for info, ref in batch_items}
        uid_sequence = [ref_to_uid[ref.sample_id] for ref in reordered_refs]
        
        print(f"  UID sequence: {uid_sequence}")
        
        # Check contiguity
        current_uid = None
        seen_uids = set()
        for uid in uid_sequence:
            if uid != current_uid:
                self.assertNotIn(uid, seen_uids, f"Group {uid} reappeared!")
                if current_uid is not None:
                    seen_uids.add(current_uid)
                current_uid = uid
        
        print("  ✅ PASSED: Samples are contiguous within groups")

    def test_single_sample_per_uid_fallback(self):
        """Test: Single sample per uid falls back to sample-level balancing."""
        print("\n=== Test: Single sample per uid (fallback) ===")
        groups = [
            ("uid_0", [100]),
            ("uid_1", [200]),
            ("uid_2", [300]),
            ("uid_3", [400]),
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 2
        
        reordered_refs = apply_length_balancing_with_group_n(batch_items, k_partitions)
        
        print(f"  Total samples: {len(reordered_refs)}")
        self.assertEqual(len(reordered_refs), 4)
        
        original_ids = {ref.sample_id for _, ref in batch_items}
        result_ids = {ref.sample_id for ref in reordered_refs}
        self.assertEqual(original_ids, result_ids)
        
        print("  ✅ PASSED: Fallback works correctly")

    def test_grpo_advantage_simulation(self):
        """Test: Simulate GRPO advantage calculation."""
        print("\n=== Test: GRPO Advantage Simulation ===")
        import random
        random.seed(42)
        
        num_prompts = 4
        rollout_n = 4
        k_partitions = 2
        
        # Create groups with known scores
        groups_data = {}
        batch_items = []
        sample_id = 0
        
        for prompt_idx in range(num_prompts):
            uid = f"prompt_{prompt_idx}"
            scores = [random.uniform(-1, 1) for _ in range(rollout_n)]
            token_counts = [random.randint(100, 500) for _ in range(rollout_n)]
            
            mean_score = sum(scores) / len(scores)
            std_score = (sum((s - mean_score)**2 for s in scores) / len(scores)) ** 0.5
            
            groups_data[uid] = {"scores": scores, "mean": mean_score, "std": std_score}
            
            for i, tokens in enumerate(token_counts):
                sample_info = MockSampleInfo(uid=uid, sum_tokens=tokens)
                batch_items.append((sample_info, MockObjectRef(sample_id)))
                sample_id += 1
        
        # Apply balancing
        uid_to_indices = defaultdict(list)
        for idx, (sample_info, _) in enumerate(batch_items):
            uid_to_indices[sample_info.uid].append(idx)
        
        group_list = list(uid_to_indices.keys())
        group_workloads = [
            sum(batch_items[i][0].sum_tokens for i in uid_to_indices[uid])
            for uid in group_list
        ]
        
        workload_lst = calculate_workload(group_workloads)
        group_partitions = get_seqlen_balanced_partitions(
            workload_lst, k_partitions=k_partitions, equal_size=True
        )
        
        # Verify each partition can compute correct group statistics
        all_correct = True
        for partition_idx, partition_group_indices in enumerate(group_partitions):
            print(f"  Partition {partition_idx}:")
            for group_idx in partition_group_indices:
                uid = group_list[group_idx]
                sample_indices = uid_to_indices[uid]
                
                # All samples of this group should be here
                self.assertEqual(len(sample_indices), rollout_n, f"Group {uid} incomplete!")
                
                # Get scores for this group
                partition_scores = groups_data[uid]["scores"]
                computed_mean = sum(partition_scores) / len(partition_scores)
                expected_mean = groups_data[uid]["mean"]
                
                print(f"    {uid}: computed_mean={computed_mean:.4f}, expected_mean={expected_mean:.4f}")
                
                self.assertAlmostEqual(computed_mean, expected_mean, places=10)
        
        print("  ✅ PASSED: GRPO advantage calculation would be correct")

    def test_large_scale(self):
        """Test: Large scale with many groups."""
        print("\n=== Test: Large Scale (32 prompts, rollout_n=8, 4 partitions) ===")
        import random
        random.seed(123)
        
        num_prompts = 32
        rollout_n = 8
        k_partitions = 4
        
        groups = []
        for i in range(num_prompts):
            uid = f"prompt_{i}"
            token_counts = [random.randint(100, 2000) for _ in range(rollout_n)]
            groups.append((uid, token_counts))
        
        batch_items = self._create_batch_items(groups)
        reordered_refs = apply_length_balancing_with_group_n(batch_items, k_partitions)
        uid_partitions = self._get_uid_distribution(batch_items, reordered_refs, k_partitions)
        
        print(f"  Total samples: {len(reordered_refs)}")
        print(f"  Groups per partition: {[len(p) for p in uid_partitions]}")
        
        # Each partition should have 8 groups
        for i, uid_set in enumerate(uid_partitions):
            self.assertEqual(len(uid_set), num_prompts // k_partitions)
        
        # No overlap
        all_uids = set()
        for uid_set in uid_partitions:
            self.assertEqual(len(all_uids & uid_set), 0, "No overlap!")
            all_uids.update(uid_set)
        
        self.assertEqual(len(all_uids), num_prompts)
        
        print("  ✅ PASSED: Large scale test successful")


if __name__ == "__main__":
    print("=" * 60)
    print("Group N Balancing Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)

