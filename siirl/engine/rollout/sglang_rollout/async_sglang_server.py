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

from siirl.engine.rollout.async_server import AsyncServerBase
from siirl.global_config.params.model_args import ActorRolloutRefArguments
from siirl.engine.rollout.sglang_rollout import SGLangRollout
logger = logging.getLogger(__file__)


class AsyncSglangServer(AsyncServerBase):
    def __init__(self, config: ActorRolloutRefArguments, spmd_engine: SGLangRollout, zmq_addresses:List):
        super().__init__()
        self.config = config.rollout
        self.workers_zmq = []
        self.master_worker_zmq = None
        self.engine = spmd_engine
        self.zmq_addresses = zmq_addresses
        
    async def init_engine(self):
        self.context = zmq.Context()
        for zmq_address in self.zmq_addresses:
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
        return await self.engine.generate(prompt_ids, sampling_params, request_id)

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
