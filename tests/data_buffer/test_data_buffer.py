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
import numpy as np
from tensordict import TensorDict
from typing import Optional, Dict, Any

# Assume siirl is in PYTHONPATH
from siirl.workers.databuffer import DataBuffer
from siirl.workers.databuffer import DataProto


def compare_dataprotos(dp1: Optional[DataProto], dp2: Optional[DataProto], check_meta=True) -> bool:
    """A more robust DataProto comparison function for testing."""
    if dp1 is None and dp2 is None:
        return True
    if dp1 is None or dp2 is None:
        return False

    if check_meta and dp1.meta_info != dp2.meta_info:
        return False

    # Compare batch (TensorDict)
    batch1_is_none = dp1.batch is None or not dp1.batch.keys()
    batch2_is_none = dp2.batch is None or not dp2.batch.keys()

    if batch1_is_none != batch2_is_none:
        return False

    if not batch1_is_none:
        if dp1.batch.batch_size != dp2.batch.batch_size:
            return False
        if set(dp1.batch.keys()) != set(dp2.batch.keys()):
            return False
        for key in dp1.batch.keys():
            if not torch.equal(dp1.batch[key], dp2.batch[key]):
                return False

    # Compare non_tensor_batch (Dict[str, np.ndarray])
    if set(dp1.non_tensor_batch.keys()) != set(dp2.non_tensor_batch.keys()):
        return False
    for key in dp1.non_tensor_batch.keys():
        if not np.array_equal(dp1.non_tensor_batch[key], dp2.non_tensor_batch[key]):
            return False

    return True


class TestDataBuffer(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the new logic in DataBuffer:
    - put: Executes concat under specific conditions.
    - get: Returns either an ObjectRef or a sub-shard of a DataProto.
    """

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(num_cpus=2, ignore_reinit_error=True, logging_level="error")

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        """Create a new, clean DataBuffer actor for each test."""
        # Use a unique name to avoid conflicts between tests
        actor_name = f"TestBuffer_{asyncio.get_running_loop().time()}"
        # Dynamically create an Actor class from the DataBuffer class
        RemoteDataBuffer = ray.remote(DataBuffer)
        self.buffer = RemoteDataBuffer.options(name=actor_name, lifetime="detached").remote(buffer_id=0)  # type: ignore
        # Ensure the actor has started
        self.assertTrue(await self.buffer.put.remote("__init__", self._create_dp({})))

    async def asyncTearDown(self):
        """Destroy the actor after each test."""
        ray.kill(self.buffer, no_restart=True)
        # Wait briefly to ensure the actor is properly cleaned up
        await asyncio.sleep(0.1)

    def _create_dp(self, tensor_data: Dict[str, Any], non_tensor_data: Optional[Dict[str, Any]] = None, meta_info: Optional[Dict] = None) -> DataProto:
        """A simple helper function to create a DataProto for testing."""
        tensors = {k: torch.tensor(v) for k, v in tensor_data.items()} if tensor_data else {}
        non_tensors = {k: np.array(v) for k, v in non_tensor_data.items()} if non_tensor_data else {}

        batch_size = []
        if tensors:
            # Use the shape of the first tensor to determine the batch size
            batch_size = [tensors[list(tensors.keys())[0]].shape[0]]

        td = TensorDict(tensors, batch_size=batch_size) if tensors else None
        return DataProto(batch=td, non_tensor_batch=non_tensors, meta_info=meta_info or {})

    # === `put` method tests ===

    async def test_put_concat_two_dataprotos(self):
        """Core test: Verify the concat operation of put on two DataProtos."""
        key = "concat_key"
        dp1 = self._create_dp({"data": [[1, 1], [2, 2]]}, meta_info={"id": "first_put"})
        dp2 = self._create_dp({"data": [[3, 3]]}, meta_info={"id": "second_put"})

        # First put
        self.assertTrue(await self.buffer.put.remote(key, dp1))
        # Second put, should trigger concat
        self.assertTrue(await self.buffer.put.remote(key, dp2))

        # Retrieve the result for verification
        retrieved_dp = await self.buffer.get.remote(key, 0, 1)

        # Expected concat result
        expected_tensors = {"data": [[1, 1], [2, 2], [3, 3]]}
        # DataProto.concat uses the meta_info of the first element in the list.
        # In DataBuffer, it's `concat([existing_value, value])`, so the meta_info should be from dp1.
        expected_meta = {"id": "first_put"}
        expected_dp = self._create_dp(expected_tensors, meta_info=expected_meta)

        self.assertTrue(compare_dataprotos(retrieved_dp, expected_dp), "DataProtos after concat do not match expected result")

    async def test_put_replace_if_existing_is_objectref(self):
        """Test: If the original value is an ObjectRef, putting a DataProto should perform a replacement."""
        key = "replace_ref_key"
        dp_for_ref = self._create_dp({"ref_data": [10]})
        obj_ref = ray.put(dp_for_ref)

        # Store the wrapped ObjectRef ([obj_ref])
        self.assertTrue(await self.buffer.put.remote(key, [obj_ref]))

        dp_new = self._create_dp({"new_data": [20]})
        # Store a DataProto, which should replace the previous ObjectRef
        self.assertTrue(await self.buffer.put.remote(key, dp_new))

        retrieved_dp = await self.buffer.get.remote(key, 0, 1)
        self.assertIsInstance(retrieved_dp, DataProto)
        self.assertTrue(compare_dataprotos(retrieved_dp, dp_new))

    async def test_put_replace_if_new_value_is_objectref(self):
        """Test: If the new value is an ObjectRef, it should always perform a replacement."""
        key = "replace_with_ref_key"
        dp_initial = self._create_dp({"initial_data": [30]})
        self.assertTrue(await self.buffer.put.remote(key, dp_initial))  # Store a DataProto

        dp_for_ref = self._create_dp({"ref_data": [40]})
        obj_ref = ray.put(dp_for_ref)
        # Store a wrapped ObjectRef, which should replace the previous DataProto
        self.assertTrue(await self.buffer.put.remote(key, [obj_ref]))

        retrieved_ref = await self.buffer.get.remote(key, 0, 1)
        # Now get should return the ObjectRef directly
        self.assertIsInstance(retrieved_ref, ray.ObjectRef, f"Expected ObjectRef, but got {type(retrieved_ref)}")

        retrieved_dp = await retrieved_ref  # type: ignore
        self.assertTrue(compare_dataprotos(retrieved_dp, dp_for_ref))

    # === `get` method tests ===

    async def test_get_returns_objectref_if_stored(self):
        """Test: When an ObjectRef is stored, get returns it directly."""
        key = "get_ref_key"
        dp = self._create_dp({"data": [101]})
        obj_ref = ray.put(dp)
        # Store the wrapped ObjectRef
        self.assertTrue(await self.buffer.put.remote(key, [obj_ref]))

        retrieved_item = await self.buffer.get.remote(key)
        self.assertIsInstance(retrieved_item, ray.ObjectRef, f"Expected ObjectRef, but got {type(retrieved_item)}")
        self.assertEqual(retrieved_item, obj_ref)

    async def test_get_reshards_stored_dataprotos_chunk(self):
        """Test: When a DataProto (shard) is stored, get returns a re-sliced sub-shard."""
        key = "get_reshard_key"
        # Simulate a shard stored in the buffer
        dp_chunk = self._create_dp({"data": [1, 2, 3, 4, 5, 6]}, meta_info={"id": "chunk_1"})
        self.assertTrue(await self.buffer.put.remote(key, dp_chunk))

        # Target DAG worker group size is 3
        target_world_size = 3

        # Request sub-shard for rank 0
        sub_chunk_0 = await self.buffer.get.remote(key, 0, target_world_size)
        expected_0 = self._create_dp({"data": [1, 2]}, meta_info={"id": "chunk_1"})
        self.assertTrue(compare_dataprotos(sub_chunk_0, expected_0))

        # Request sub-shard for rank 1
        sub_chunk_1 = await self.buffer.get.remote(key, 1, target_world_size)
        expected_1 = self._create_dp({"data": [3, 4]}, meta_info={"id": "chunk_1"})
        self.assertTrue(compare_dataprotos(sub_chunk_1, expected_1))

        # Request sub-shard for rank 2
        sub_chunk_2 = await self.buffer.get.remote(key, 2, target_world_size)
        expected_2 = self._create_dp({"data": [5, 6]}, meta_info={"id": "chunk_1"})
        self.assertTrue(compare_dataprotos(sub_chunk_2, expected_2))

    async def test_get_returns_none_for_nonexistent_key(self):
        """Test: When the key does not exist, get returns None."""
        retrieved_item = await self.buffer.get.remote("non_existent_key")
        self.assertIsNone(retrieved_item)

    # === `pop` method tests ===

    async def test_pop_removes_and_returns_dataprotos(self):
        """Test: pop can remove and return a DataProto."""
        key = "pop_dp_key"
        dp = self._create_dp({"data": [1, 2, 3]})

        # Store data
        await self.buffer.put.remote(key, dp)

        # Confirm data exists
        self.assertIsNotNone(await self.buffer.get.remote(key))

        # pop operation
        popped_item = await self.buffer.pop.remote(key)
        self.assertIsInstance(popped_item, DataProto)
        self.assertTrue(compare_dataprotos(popped_item, dp))

        # Confirm data has been removed
        self.assertIsNone(await self.buffer.get.remote(key))

    async def test_pop_removes_and_returns_objectref(self):
        """Test: pop can remove and return an ObjectRef."""
        key = "pop_ref_key"
        dp_for_ref = self._create_dp({"ref_data": [100]})
        obj_ref = ray.put(dp_for_ref)

        # Store the wrapped ObjectRef
        await self.buffer.put.remote(key, [obj_ref])

        # Confirm data exists
        self.assertIsNotNone(await self.buffer.get.remote(key))

        # pop operation
        popped_item = await self.buffer.pop.remote(key)
        self.assertIsInstance(popped_item, ray.ObjectRef)
        self.assertEqual(popped_item, obj_ref)

        # Confirm data has been removed
        self.assertIsNone(await self.buffer.get.remote(key))

    async def test_pop_returns_none_for_nonexistent_key(self):
        """Test: When the key does not exist, pop returns None."""
        popped_item = await self.buffer.pop.remote("non_existent_key_for_pop")
        self.assertIsNone(popped_item)


if __name__ == "__main__":
    unittest.main()
