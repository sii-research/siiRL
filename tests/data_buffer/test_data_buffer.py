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

import unittest
import asyncio
import ray
import torch
import uuid
from tensordict import TensorDict
from typing import Dict, Any, List

# Imports from the refactored implementation
from siirl.data_coordinator.data_buffer import init_data_coordinator
from siirl.data_coordinator.sample import SampleInfo


class TestDataCoordinator(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the new DataCoordinator/DataBuffer architecture.
    """

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True, logging_level="error")

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        """Create a new, clean DataCoordinator system for each test."""
        # Use force_local=True for single-node unit testing
        self.coordinator = init_data_coordinator(num_buffers=2, force_local=True)
        # Ensure the actor has started and is ready
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 0)

    async def asyncTearDown(self):
        """Destroy the actor after each test."""
        # The coordinator is a detached actor, so we must manually kill it.
        ray.kill(self.coordinator, no_restart=True)
        # Allow some time for cleanup
        await asyncio.sleep(0.1)

    def _create_mock_sample(self, content_id: int) -> TensorDict:
        """Helper to create a sample with identifiable content."""
        return TensorDict({"data": torch.tensor([[content_id]])}, batch_size=[1])

    def _create_mock_sample_info(self, tokens: int, group: int = 0) -> SampleInfo:
        """Helper to create a SampleInfo object."""
        return SampleInfo(
            agent_group=group,
            sum_tokens=tokens,
            prompt_length=tokens,
            response_length=0,
            uid=uuid.uuid4().int
        )

    # === Test Cases ===

    async def test_put_increases_size(self):
        """Test that calling `put` increases the coordinator's queue size."""
        initial_size = await self.coordinator.get_valid_size.remote()
        self.assertEqual(initial_size, 0)

        sample_ref = ray.put(self._create_mock_sample(1))
        sample_info = self._create_mock_sample_info(tokens=128)
        
        await self.coordinator.put.remote(sample_info, sample_ref)
        
        new_size = await self.coordinator.get_valid_size.remote()
        self.assertEqual(new_size, 1)

    async def test_get_batch_simple(self):
        """Test basic `get_batch` functionality and data integrity."""
        # 1. Put a sample
        sample_data = self._create_mock_sample(101)
        sample_ref = ray.put(sample_data)
        sample_info = self._create_mock_sample_info(tokens=128)
        await self.coordinator.put.remote(sample_info, sample_ref)
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 1)

        # 2. Get a batch
        batch_refs_or_values = await self.coordinator.get_batch.remote(batch_size=1)
        self.assertEqual(len(batch_refs_or_values), 1)
        
        # 3. Verify queue size decreases
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 0)

        # 4. Verify data integrity
        retrieved_item = batch_refs_or_values[0]
        if isinstance(retrieved_item, ray.ObjectRef):
            retrieved_item = ray.get(retrieved_item)
        
        self.assertTrue(torch.equal(retrieved_item.get("data"), sample_data.get("data")))

    async def test_get_batch_insufficient_samples(self):
        """Test that `get_batch` returns an empty list when samples are insufficient."""
        # Queue is empty
        batch = await self.coordinator.get_batch.remote(batch_size=1)
        self.assertEqual(len(batch), 0)

        # Queue has 1, but we request 2
        sample_ref = ray.put(self._create_mock_sample(1))
        sample_info = self._create_mock_sample_info(tokens=128)
        await self.coordinator.put.remote(sample_info, sample_ref)
        
        batch = await self.coordinator.get_batch.remote(batch_size=2)
        self.assertEqual(len(batch), 0)
        # Ensure the queue was not modified
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 1)

    async def test_get_batch_fifo_order(self):
        """Test that `get_batch` respects FIFO order without a filter."""
        # Put 3 samples with identifiable content IDs
        samples_put = [self._create_mock_sample(i) for i in [10, 20, 30]]
        for sample in samples_put:
            sample_ref = ray.put(sample)
            sample_info = self._create_mock_sample_info(tokens=128)
            await self.coordinator.put.remote(sample_info, sample_ref)

        # Get a batch of 2
        batch_refs_or_values = await self.coordinator.get_batch.remote(batch_size=2)
        retrieved_data = [ray.get(item) if isinstance(item, ray.ObjectRef) else item for item in batch_refs_or_values]
        
        # Verify the first two samples were returned
        self.assertTrue(torch.equal(retrieved_data[0].get("data"), samples_put[0].get("data")))
        self.assertTrue(torch.equal(retrieved_data[1].get("data"), samples_put[1].get("data")))
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 1)

    async def test_get_batch_with_filter(self):
        """Test that `get_batch` correctly applies the filter_plugin."""
        # Put 4 samples with different token counts
        sample_infos = [
            self._create_mock_sample_info(tokens=100), # Should be filtered out
            self._create_mock_sample_info(tokens=600), # Should be selected
            self._create_mock_sample_info(tokens=200), # Should be filtered out
            self._create_mock_sample_info(tokens=700), # Should be selected
        ]
        samples_put = [self._create_mock_sample(info.sum_tokens) for info in sample_infos]

        for info, data in zip(sample_infos, samples_put):
            sample_ref = ray.put(data)
            await self.coordinator.put.remote(info, sample_ref)
        
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 4)
        
        # Define a filter that only accepts samples with >= 512 tokens
        def long_sample_filter(sample_info: SampleInfo) -> bool:
            return sample_info.sum_tokens >= 512

        # Request a batch of 2 with the filter
        batch_refs_or_values = await self.coordinator.get_batch.remote(
            batch_size=2, filter_plugin=long_sample_filter
        )
        self.assertEqual(len(batch_refs_or_values), 2)
        
        # Verify the correct samples were returned (600 and 700)
        retrieved_data = [ray.get(item) if isinstance(item, ray.ObjectRef) else item for item in batch_refs_or_values]
        retrieved_tokens = sorted([item.get("data").item() for item in retrieved_data])
        self.assertEqual(retrieved_tokens, [600, 700])

        # Verify that the filtered-out samples remain in the queue
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 2)


    async def test_get_batch_with_node_affinity_filter(self):
        """
        Test using the filter_plugin to achieve node-affinity scheduling.
        This simulates a trainer on 'node_A' preferentially pulling data
        that was also produced on 'node_A'.
        """
        # 1. Simulate data coming from two different nodes
        # Samples from node_A
        for i in range(3):
            info = self._create_mock_sample_info(tokens=128, group=i)
            info.node_id = "node_A" # Manually set node_id for testing
            data = self._create_mock_sample(content_id=100 + i)
            await self.coordinator.put.remote(info, ray.put(data))

        # Samples from node_B
        for i in range(2):
            info = self._create_mock_sample_info(tokens=256, group=i)
            info.node_id = "node_B" # Manually set node_id for testing
            data = self._create_mock_sample(content_id=200 + i)
            await self.coordinator.put.remote(info, ray.put(data))

        self.assertEqual(await self.coordinator.get_valid_size.remote(), 5)

        # 2. Create a filter factory to generate an affinity filter
        # This closure captures the desired local_node_id.
        def create_affinity_filter(local_node_id: str):
            def affinity_filter(sample_info: SampleInfo) -> bool:
                return sample_info.node_id == local_node_id
            return affinity_filter

        # 3. Simulate a Trainer on 'node_A' pulling its local data
        trainer_on_node_a_filter = create_affinity_filter("node_A")
        
        local_batch = await self.coordinator.get_batch.remote(
            batch_size=3, filter_plugin=trainer_on_node_a_filter
        )
        self.assertEqual(len(local_batch), 3)

        # Verify that we got all the data from node_A
        retrieved_data = ray.get(local_batch)
        retrieved_ids = sorted([d.get("data").item() for d in retrieved_data])
        self.assertEqual(retrieved_ids, [100, 101, 102])
        
        # 4. The coordinator should now only contain data from node_B
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 2)

        # 5. Now, the trainer on 'node_A' can fetch remote data if needed
        # (by inverting the filter or using a different one)
        trainer_on_node_b_filter = create_affinity_filter("node_B")
        remote_batch = await self.coordinator.get_batch.remote(
            batch_size=2, filter_plugin=trainer_on_node_b_filter
        )
        self.assertEqual(len(remote_batch), 2)
        retrieved_data = ray.get(remote_batch)
        retrieved_ids = sorted([d.get("data").item() for d in retrieved_data])
        self.assertEqual(retrieved_ids, [200, 201])


if __name__ == "__main__":
    unittest.main()
