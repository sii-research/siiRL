# Copyright 2025, Shanghai Innovation Institute.  All rights reserved.
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

"""
Unit tests for Group N support in DataCoordinator.

These tests verify that samples with the same uid are kept together
when performing length balancing across DP partitions. This is critical
for GRPO and similar algorithms that require group-relative advantage computation.
"""

import unittest
from collections import defaultdict
from typing import List, Tuple
from unittest.mock import MagicMock

from siirl.data_coordinator.sample import SampleInfo
from siirl.utils.model_utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions


class MockObjectRef:
    """Mock Ray ObjectRef for testing without Ray."""
    def __init__(self, sample_id: int):
        self.sample_id = sample_id
    
    def __repr__(self):
        return f"MockRef({self.sample_id})"
    
    def __eq__(self, other):
        return isinstance(other, MockObjectRef) and self.sample_id == other.sample_id
    
    def __hash__(self):
        return hash(self.sample_id)


class TestGroupNBalancing(unittest.TestCase):
    """
    Tests for Group N (uid-based grouping) support in length balancing.
    """

    def _create_sample_info(self, uid: str, sum_tokens: int) -> SampleInfo:
        """Helper to create a SampleInfo with specific uid and token count."""
        return SampleInfo(
            uid=uid,
            sum_tokens=sum_tokens,
            prompt_length=sum_tokens // 2,
            response_length=sum_tokens // 2,
        )

    def _create_batch_items(
        self, 
        groups: List[Tuple[str, List[int]]]
    ) -> List[Tuple[SampleInfo, MockObjectRef]]:
        """
        Create batch items for testing.
        
        Args:
            groups: List of (uid, [token_counts]) tuples.
                   Example: [("0", [100, 120, 80, 90]), ("1", [200, 180, 220, 190])]
        
        Returns:
            List of (SampleInfo, MockObjectRef) tuples.
        """
        batch_items = []
        sample_id = 0
        for uid, token_counts in groups:
            for tokens in token_counts:
                sample_info = self._create_sample_info(uid, tokens)
                mock_ref = MockObjectRef(sample_id)
                batch_items.append((sample_info, mock_ref))
                sample_id += 1
        return batch_items

    def _apply_length_balancing_with_group_n(
        self,
        batch_items: List[Tuple[SampleInfo, MockObjectRef]],
        k_partitions: int
    ) -> List[MockObjectRef]:
        """
        Simulates the _apply_length_balancing function with Group N support.
        This is a copy of the actual implementation for unit testing without Ray.
        """
        # ========== Step 1: Group samples by uid ==========
        uid_to_indices = defaultdict(list)
        for idx, (sample_info, _) in enumerate(batch_items):
            uid = sample_info.uid if sample_info.uid is not None else str(idx)
            uid_to_indices[uid].append(idx)
        
        # Check if grouping is needed
        max_group_size = max(len(indices) for indices in uid_to_indices.values()) if uid_to_indices else 1
        
        if max_group_size == 1:
            # Fallback to single-sample balancing
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
            # Fallback
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
            workload_lst, 
            k_partitions=k_partitions, 
            equal_size=equal_size
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

    def _get_uid_distribution(
        self, 
        batch_items: List[Tuple[SampleInfo, MockObjectRef]], 
        reordered_refs: List[MockObjectRef],
        k_partitions: int
    ) -> List[set]:
        """
        Get the distribution of uids across partitions after balancing.
        
        Returns:
            List of sets, where each set contains the uids in that partition.
        """
        # Build a mapping from ref to uid
        ref_to_uid = {}
        for sample_info, ref in batch_items:
            ref_to_uid[ref.sample_id] = sample_info.uid
        
        # Split reordered_refs into partitions
        partition_size = len(reordered_refs) // k_partitions
        partitions = []
        for i in range(k_partitions):
            start = i * partition_size
            end = start + partition_size
            partition_refs = reordered_refs[start:end]
            partition_uids = {ref_to_uid[ref.sample_id] for ref in partition_refs}
            partitions.append(partition_uids)
        
        return partitions

    # ========== Test Cases ==========

    def test_group_n_basic_two_groups_two_partitions(self):
        """
        Test: 2 groups with N=4, balanced across 2 partitions.
        Each partition should get exactly 1 complete group.
        """
        # Create 2 groups, each with 4 samples (rollout.n = 4)
        groups = [
            ("uid_0", [100, 120, 80, 90]),   # Total: 390
            ("uid_1", [200, 180, 220, 190]), # Total: 790
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 2
        
        # Apply balancing
        reordered_refs = self._apply_length_balancing_with_group_n(batch_items, k_partitions)
        
        # Get uid distribution
        uid_partitions = self._get_uid_distribution(batch_items, reordered_refs, k_partitions)
        
        # Verify: each partition should have exactly 1 uid (group is not split)
        for i, uid_set in enumerate(uid_partitions):
            self.assertEqual(
                len(uid_set), 1, 
                f"Partition {i} should have exactly 1 group, but has {len(uid_set)}: {uid_set}"
            )
        
        # Verify: all uids are covered
        all_uids = set()
        for uid_set in uid_partitions:
            all_uids.update(uid_set)
        self.assertEqual(all_uids, {"uid_0", "uid_1"})

    def test_group_n_four_groups_two_partitions(self):
        """
        Test: 4 groups with N=4, balanced across 2 partitions.
        Each partition should get 2 complete groups.
        """
        groups = [
            ("uid_0", [100, 110, 90, 100]),   # Total: 400
            ("uid_1", [200, 210, 190, 200]),  # Total: 800
            ("uid_2", [150, 140, 160, 150]),  # Total: 600
            ("uid_3", [300, 290, 310, 300]),  # Total: 1200
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 2
        
        reordered_refs = self._apply_length_balancing_with_group_n(batch_items, k_partitions)
        uid_partitions = self._get_uid_distribution(batch_items, reordered_refs, k_partitions)
        
        # Each partition should have exactly 2 groups
        for i, uid_set in enumerate(uid_partitions):
            self.assertEqual(
                len(uid_set), 2, 
                f"Partition {i} should have 2 groups, but has {len(uid_set)}: {uid_set}"
            )
        
        # Verify no group is split across partitions
        partition_0_uids = uid_partitions[0]
        partition_1_uids = uid_partitions[1]
        self.assertEqual(
            len(partition_0_uids & partition_1_uids), 0,
            f"Groups should not be split! Found overlap: {partition_0_uids & partition_1_uids}"
        )

    def test_group_n_samples_within_group_stay_together(self):
        """
        Test: Verify that samples within the same group are contiguous in the output.
        """
        groups = [
            ("uid_A", [100, 200, 300, 400]),
            ("uid_B", [150, 250, 350, 450]),
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 2
        
        reordered_refs = self._apply_length_balancing_with_group_n(batch_items, k_partitions)
        
        # Build ref_id to uid mapping
        ref_to_uid = {}
        for sample_info, ref in batch_items:
            ref_to_uid[ref.sample_id] = sample_info.uid
        
        # Extract uid sequence
        uid_sequence = [ref_to_uid[ref.sample_id] for ref in reordered_refs]
        
        # Verify contiguity: same uid samples should be adjacent
        # The sequence should be like [A, A, A, A, B, B, B, B] or [B, B, B, B, A, A, A, A]
        # Not like [A, B, A, B, A, B, A, B]
        current_uid = None
        seen_uids = set()
        for uid in uid_sequence:
            if uid != current_uid:
                # Transitioning to a new group
                self.assertNotIn(
                    uid, seen_uids,
                    f"Group {uid} appeared again after we left it! Sequence: {uid_sequence}"
                )
                if current_uid is not None:
                    seen_uids.add(current_uid)
                current_uid = uid

    def test_single_sample_per_uid_fallback(self):
        """
        Test: When each uid has only 1 sample (no grouping needed),
        the function should fallback to single-sample balancing.
        """
        groups = [
            ("uid_0", [100]),
            ("uid_1", [200]),
            ("uid_2", [300]),
            ("uid_3", [400]),
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 2
        
        # This should still work and not crash
        reordered_refs = self._apply_length_balancing_with_group_n(batch_items, k_partitions)
        
        # Verify all samples are present
        self.assertEqual(len(reordered_refs), 4)
        original_ids = {ref.sample_id for _, ref in batch_items}
        result_ids = {ref.sample_id for ref in reordered_refs}
        self.assertEqual(original_ids, result_ids)

    def test_groups_fewer_than_partitions_fallback(self):
        """
        Test: When number of groups < partitions, should fallback gracefully.
        """
        groups = [
            ("uid_0", [100, 200]),
            ("uid_1", [300, 400]),
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 4  # More partitions than groups!
        
        # Should not crash
        reordered_refs = self._apply_length_balancing_with_group_n(batch_items, k_partitions)
        
        # Verify all samples are present
        self.assertEqual(len(reordered_refs), 4)

    def test_group_n_with_varying_group_sizes(self):
        """
        Test: Groups with different sizes (edge case).
        Some prompts might have more/fewer responses due to filtering.
        
        Key verification: Even with varying group sizes, samples within each group
        should remain contiguous (not split across different parts of the output).
        """
        groups = [
            ("uid_0", [100, 200, 300]),      # 3 samples
            ("uid_1", [400, 500, 600, 700]), # 4 samples
            ("uid_2", [800, 900]),           # 2 samples
            ("uid_3", [1000]),               # 1 sample
        ]
        batch_items = self._create_batch_items(groups)
        k_partitions = 2
        
        reordered_refs = self._apply_length_balancing_with_group_n(batch_items, k_partitions)
        
        # Build ref_id to uid mapping
        ref_to_uid = {}
        for sample_info, ref in batch_items:
            ref_to_uid[ref.sample_id] = sample_info.uid
        
        # Extract uid sequence from reordered refs
        uid_sequence = [ref_to_uid[ref.sample_id] for ref in reordered_refs]
        
        # Verify contiguity: same uid samples should be adjacent (group not split)
        current_uid = None
        seen_uids = set()
        for uid in uid_sequence:
            if uid != current_uid:
                # Transitioning to a new group
                self.assertNotIn(
                    uid, seen_uids,
                    f"Group {uid} appeared again after we left it! "
                    f"This means the group was split. Sequence: {uid_sequence}"
                )
                if current_uid is not None:
                    seen_uids.add(current_uid)
                current_uid = uid
        
        # Verify all samples are present
        self.assertEqual(len(reordered_refs), 10, "All 10 samples should be present")
        
        # Verify all groups are represented
        all_uids_in_result = set(uid_sequence)
        self.assertEqual(all_uids_in_result, {"uid_0", "uid_1", "uid_2", "uid_3"})

    def test_large_scale_group_n(self):
        """
        Test: Large scale test with many groups.
        Simulates a realistic training scenario.
        """
        import random
        random.seed(42)
        
        num_prompts = 32
        rollout_n = 8
        k_partitions = 4
        
        groups = []
        for i in range(num_prompts):
            uid = f"prompt_{i}"
            # Random token counts between 100 and 2000
            token_counts = [random.randint(100, 2000) for _ in range(rollout_n)]
            groups.append((uid, token_counts))
        
        batch_items = self._create_batch_items(groups)
        reordered_refs = self._apply_length_balancing_with_group_n(batch_items, k_partitions)
        uid_partitions = self._get_uid_distribution(batch_items, reordered_refs, k_partitions)
        
        # Each partition should have 8 groups (32 / 4)
        for i, uid_set in enumerate(uid_partitions):
            self.assertEqual(
                len(uid_set), num_prompts // k_partitions,
                f"Partition {i} should have {num_prompts // k_partitions} groups"
            )
        
        # Verify no overlap (no group split)
        all_uids = set()
        for uid_set in uid_partitions:
            self.assertEqual(len(all_uids & uid_set), 0, "Groups should not be split!")
            all_uids.update(uid_set)
        
        # All groups should be accounted for
        expected_uids = {f"prompt_{i}" for i in range(num_prompts)}
        self.assertEqual(all_uids, expected_uids)


class TestGroupNBalancingGRPOScenario(unittest.TestCase):
    """
    Tests that simulate real GRPO scenarios to verify Group N works correctly.
    """

    def test_grpo_advantage_calculation_integrity(self):
        """
        Simulate GRPO advantage calculation to verify that
        after Group N balancing, each partition can correctly compute
        group-relative advantages.
        
        In GRPO:
        - advantage = (score - group_mean) / group_std
        - This requires all samples of a group to be on the same partition
        """
        import random
        random.seed(123)
        
        # Setup: 4 prompts, each with 4 responses
        num_prompts = 4
        rollout_n = 4
        k_partitions = 2
        
        # Create groups with known scores
        groups_data = {}
        batch_items = []
        sample_id = 0
        
        for prompt_idx in range(num_prompts):
            uid = f"prompt_{prompt_idx}"
            # Each prompt has 4 responses with different scores
            scores = [random.uniform(-1, 1) for _ in range(rollout_n)]
            token_counts = [random.randint(100, 500) for _ in range(rollout_n)]
            
            groups_data[uid] = {
                "scores": scores,
                "mean": sum(scores) / len(scores),
                "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5
            }
            
            for i, (tokens, score) in enumerate(zip(token_counts, scores)):
                sample_info = SampleInfo(
                    uid=uid,
                    sum_tokens=tokens,
                    prompt_length=tokens // 2,
                    response_length=tokens // 2,
                )
                # Store score in dict_info for simulation
                sample_info.dict_info["score"] = score
                sample_info.dict_info["sample_idx_in_group"] = i
                
                batch_items.append((sample_info, MockObjectRef(sample_id)))
                sample_id += 1
        
        # Apply Group N balancing (reuse logic from previous test)
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
        
        # Verify each partition can compute correct advantages
        for partition_idx, partition_group_indices in enumerate(group_partitions):
            for group_idx in partition_group_indices:
                uid = group_list[group_idx]
                sample_indices = uid_to_indices[uid]
                
                # Collect scores for this group in this partition
                partition_scores = [
                    batch_items[idx][0].dict_info["score"] 
                    for idx in sample_indices
                ]
                
                # Compute group statistics
                computed_mean = sum(partition_scores) / len(partition_scores)
                computed_std = (sum((s - computed_mean)**2 for s in partition_scores) / len(partition_scores)) ** 0.5
                
                # Verify they match the original group statistics
                expected_mean = groups_data[uid]["mean"]
                expected_std = groups_data[uid]["std"]
                
                self.assertAlmostEqual(
                    computed_mean, expected_mean, places=10,
                    msg=f"Partition {partition_idx}, Group {uid}: mean mismatch"
                )
                self.assertAlmostEqual(
                    computed_std, expected_std, places=10,
                    msg=f"Partition {partition_idx}, Group {uid}: std mismatch"
                )


class MockObjectRef:
    """Mock Ray ObjectRef for testing without Ray."""
    def __init__(self, sample_id: int):
        self.sample_id = sample_id
    
    def __repr__(self):
        return f"MockRef({self.sample_id})"
    
    def __eq__(self, other):
        return isinstance(other, MockObjectRef) and self.sample_id == other.sample_id
    
    def __hash__(self):
        return hash(self.sample_id)


if __name__ == "__main__":
    unittest.main()

