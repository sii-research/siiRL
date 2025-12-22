# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import logging
from typing import Any, Dict, List, Tuple
import pickle
import zmq
import torch
import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse

from siirl.workers.rollout.async_server import AsyncServerBase
from siirl.utils.params.model_args import ActorRolloutRefArguments
logger = logging.getLogger(__file__)


@ray.remote(num_cpus=1)
class AsyncSglangServer(AsyncServerBase):
    def __init__(self, config: ActorRolloutRefArguments, global_rank: int, wg_prefix: str):
        super().__init__()
        self.config = config.rollout
        self.global_rank = global_rank
        self.wg_prefix = wg_prefix
        self.workers_zmq = []
        self.master_worker_zmq = None
        
    async def init_engine(self):
        if self.workers_zmq:
            # avoid init twice
            return
        actor_names = [
            actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{self.wg_prefix}_DAGWorker")
        ]
        vllm_tp_size = self.config.tensor_model_parallel_size
        def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
            fields = actor_name.split(":")
            assert len(fields) == 2, f"invalid actor name: {actor_name}"
            pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
            return pg_index, local_rank
    # sort actor names by pg_index and local_rank
        actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
        actor_names = actor_names[self.global_rank : self.global_rank + vllm_tp_size]
        matched_actors = [
            ray.get_actor(actor_name) for actor_name in actor_names
        ]
        self.context = zmq.Context()
        for worker in matched_actors:
            zmq_address = ray.get(worker.get_zeromq_address.remote())
            socket = self.context.socket(zmq.REQ)
            socket.connect(zmq_address)
            self.workers_zmq.append(socket)
        self.master_worker_zmq = self.workers_zmq[0]

    async def chat_completion(self, raw_request: Request):
        request = await raw_request.json()
        message = pickle.dumps(('chat_completion', (), {'request':request}))
        self.master_worker_zmq.send(message, zmq.DONTWAIT)
        outputs = []
        outputs.append(pickle.loads(self.master_worker_zmq.recv()))
        return JSONResponse(outputs)


    async def generate(self, prompt_ids: List[int], sampling_params: Dict[str, Any], request_id: str) -> List[int]:
        message = pickle.dumps(('generate', (), {'prompt_ids':prompt_ids, 'sampling_params':sampling_params, 'request_id':request_id}))
        self.master_worker_zmq.send(message, zmq.DONTWAIT)
        return pickle.loads(self.master_worker_zmq.recv())
        return await self.master_worker.generate.remote(prompt_ids, sampling_params, request_id)

    async def wake_up(self):
        if not self.config.free_cache_engine:
            return
        message = pickle.dumps(('wake_up', (), {}))
        for socket in self.workers_zmq:
            socket.send(message, zmq.DONTWAIT)
        for socket in self.workers_zmq:
            socket.recv()
        return

    async def sleep(self):
        if not self.config.free_cache_engine:
            return
        message = pickle.dumps(('sleep', (), {}))
        for socket in self.workers_zmq:
            socket.send(message, zmq.DONTWAIT)
        for socket in self.workers_zmq:
            socket.recv()
        return
