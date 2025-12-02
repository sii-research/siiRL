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
import torch.distributed as dist
from tensordict import TensorDict
from typing import List, Optional, Dict

from siirl.workers.databuffer.data_buffer import DataBuffer
from tests.data_buffer.test_data_buffer import compare_dataprotos


# Mock Tokenizer for testing purposes
class MockTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.padding_side = "right"


@ray.remote
class MockDAGWorker:
    """
    A mock DAGWorker that only contains the logic needed to test put/get.
    It does not load a model or execute an actual computation graph.
    """

    def __init__(self, rank: int, world_size: int, data_buffers: List[ray.actor.ActorHandle]):
        self._rank = rank
        self._world_size = world_size
        self.data_buffers = data_buffers
        self.tokenizer = MockTokenizer()

        # Mock the distributed setup for the actor
        dist.init_process_group(
            backend="gloo",  # Use gloo for CPU-based testing
            init_method=f"tcp://127.0.0.1:29501",
            world_size=self._world_size,
            rank=self._rank,
        )

    async def put_data_to_buffers(self, key: str, data: TensorDict, source_dp_size: int, dest_dp_size: int):
        # This is a copy of the logic from the actual DAGWorker
        data.meta_info["padding_values"] = {
            "input_ids": self.tokenizer.pad_token_id,
            "responses": self.tokenizer.pad_token_id,
            "labels": -100,
            "attention_mask": 0,
            "response_mask": 0,
        }
        data.meta_info["padding_side"] = self.tokenizer.padding_side

        if source_dp_size == dest_dp_size:
            obj_ref = ray.put(data)
            await self.data_buffers[0].put.remote(key, [obj_ref])
        else:
            num_physical_buffers = len(self.data_buffers)
            chunks = data.chunk(chunks=num_physical_buffers)
            put_futures = [buffer.put.remote(key, chunk) for buffer, chunk in zip(self.data_buffers, chunks)]
            await asyncio.gather(*put_futures)

    async def get_data_from_buffers(self, key: str, my_current_dp_rank: int, my_current_dp_world_size: int) -> Optional[TensorDict]:
        # This is a copy of the logic from the actual DAGWorker
        if not self.data_buffers:
            return None

        first_item = await self.data_buffers[0].get.remote(key, my_current_dp_rank, my_current_dp_world_size)

        if first_item is None:
            return None

        if isinstance(first_item, ray.ObjectRef):
            return await first_item

        elif isinstance(first_item, TensorDict):
            my_sub_chunks = [first_item]
            get_futures = [buffer.get.remote(key, my_current_dp_rank, my_current_dp_world_size) for buffer in self.data_buffers[1:]]
            other_sub_chunks = await asyncio.gather(*get_futures)

            for sub_chunk in other_sub_chunks:
                if not isinstance(sub_chunk, TensorDict):
                    return None
                my_sub_chunks.append(sub_chunk)

            return TensorDict.concat(my_sub_chunks)

        return None

    def get_rank(self):
        return self._rank

    def barrier(self):
        dist.barrier()


def _create_test_dp(batch_size: int, seq_len: int, meta: Optional[Dict] = None) -> TensorDict:
    """Creates a sample TensorDict for testing."""
    tensors = {
        "input_ids": torch.randint(1, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }
    td = TensorDict(tensors, batch_size=[batch_size])
    return TensorDict(batch=td, meta_info=meta or {})


class TestDAGWorkerDataFlow(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(num_cpus=8, ignore_reinit_error=True, logging_level="error")

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def test_put_get_flow_sharded(self):
        """
        Tests the data flow for sharded storage (source_dp != dest_dp).
        - A single worker (rank 0) puts a sharded TensorDict.
        - All workers get their corresponding data and reconstruct it.
        """
        num_buffers = 2
        num_workers = 4
        source_dp, dest_dp = 2, 4
        key = "sharded_key"

        # 1. Setup Actors
        buffers = [DataBuffer.remote(buffer_id=i) for i in range(num_buffers)]
        workers = [MockDAGWorker.remote(rank=i, world_size=num_workers, data_buffers=buffers) for i in range(num_workers)]
        await asyncio.sleep(1)  # Wait for actors to initialize dist group

        # 2. Prepare Data
        # Total batch size should be divisible by dest_dp and num_buffers
        total_batch_size = dest_dp * num_buffers
        full_dp = _create_test_dp(total_batch_size, seq_len=10)

        # 3. Leader (worker 0) puts data
        # Data is sharded across the 2 buffers
        await workers[0].put_data_to_buffers.remote(key, full_dp, source_dp_size=source_dp, dest_dp_size=dest_dp)

        # 4. All workers synchronize
        await asyncio.gather(*[w.barrier.remote() for w in workers])

        # 5. All workers get their data slice
        get_futures = [w.get_data_from_buffers.remote(key, my_current_dp_rank=await w.get_rank.remote(), my_current_dp_world_size=dest_dp) for w in workers]
        results = await asyncio.gather(*get_futures)

        # 6. Verify results
        # In sharded mode, data is distributed in an interleaved manner. We must manually reconstruct
        # the expected shard for each worker.
        buffer_chunks = full_dp.chunk(chunks=num_buffers)
        expected_shards_by_worker = []
        for worker_rank in range(dest_dp):
            sub_chunks_for_worker = []
            for buffer_chunk in buffer_chunks:
                # The chunk from each buffer will be re-sharded for the target worker.
                sub_sub_chunks = buffer_chunk.chunk(chunks=dest_dp)
                sub_chunks_for_worker.append(sub_sub_chunks[worker_rank])
            # The worker's final data is the concatenation of the sub-shards it receives from all buffers.
            expected_shards_by_worker.append(TensorDict.concat(sub_chunks_for_worker))

        self.assertEqual(len(results), len(expected_shards_by_worker))

        for i, result_dp in enumerate(results):
            self.assertIsNotNone(result_dp, f"Worker {i} received None")
            expected_dp = expected_shards_by_worker[i]
            self.assertTrue(compare_dataprotos(expected_dp, result_dp, check_meta=False), f"TensorDict for worker {i} does not match the expected interleaved shard. Expected size {len(expected_dp)}, got {len(result_dp)}")

        # Cleanup
        for actor in workers + buffers:
            ray.kill(actor, no_restart=True)

    async def test_put_get_flow_object_ref(self):
        """
        Tests the data flow for ObjectRef storage (source_dp == dest_dp).
        - A single worker (rank 0) puts a TensorDict, which becomes an ObjectRef.
        - All workers get and resolve the same ObjectRef.
        """
        num_buffers = 1  # Only one buffer is used in this case
        num_workers = 4
        source_dp, dest_dp = 4, 4
        key = "object_ref_key"

        # 1. Setup Actors
        buffers = [DataBuffer.remote(buffer_id=0)]
        workers = [MockDAGWorker.remote(rank=i, world_size=num_workers, data_buffers=buffers) for i in range(num_workers)]
        await asyncio.sleep(1)

        # 2. Prepare Data
        full_dp = _create_test_dp(batch_size=8, seq_len=12)

        # 3. Leader (worker 0) puts data
        await workers[0].put_data_to_buffers.remote(key, full_dp, source_dp_size=source_dp, dest_dp_size=dest_dp)

        # 4. All workers synchronize
        await asyncio.gather(*[w.barrier.remote() for w in workers])

        # 5. All workers get data and verify
        get_futures = [w.get_data_from_buffers.remote(key, my_current_dp_rank=await w.get_rank.remote(), my_current_dp_world_size=dest_dp) for w in workers]
        results = await asyncio.gather(*get_futures)

        # In the ObjectRef case, every worker gets the same data.
        for i, result_dp in enumerate(results):
            self.assertIsNotNone(result_dp, f"Worker {i} received None")
            self.assertTrue(compare_dataprotos(full_dp, result_dp, check_meta=False), f"TensorDict for worker {i} does not match the original")

        # Cleanup
        for actor in workers + buffers:
            ray.kill(actor)


if __name__ == "__main__":
    unittest.main()
