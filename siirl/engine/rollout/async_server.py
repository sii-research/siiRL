# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import socket
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type


import ray

from starlette.requests import Request

logger = logging.getLogger(__file__)


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class AsyncServerBase(ABC):
    """Base class for AsyncServer."""

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = None

    @abstractmethod
    async def chat_completion(self, raw_request: Request):
        """OpenAI chat completion API.

        API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        raise NotImplementedError

    @abstractmethod
    async def generate(self, prompt_ids: List[int], sampling_params: Dict[str, Any], request_id: str) -> List[int]:
        """Generate response ids given prompt ids.

        Args:
            prompt_ids (List[int]): prompt ids
            sampling_params (Dict[str, Any]): sampling params
            request_id (str): request id

        Returns:
            List[int]: response ids
        """
        raise NotImplementedError

    @abstractmethod
    async def init_engine(self):
        """Init async LLM engine."""
        raise NotImplementedError

    @abstractmethod
    async def wake_up(self):
        """Wake up engine to load model weights and build kv cache."""
        raise NotImplementedError

    @abstractmethod
    async def sleep(self):
        """Sleep engine to offload model weights and discard kv cache."""
        raise NotImplementedError


def async_server_class(
    rollout_backend: str, rollout_backend_module: Optional[str] = None, rollout_backend_class: Optional[str] = None
) -> Type[AsyncServerBase]:
    """Get async server class.

    Args:
        rollout_backend: str, rollout backend type (alias), should be "vllm" or "sglang".
        rollout_backend_module: Optional[str], import path of the rollout backend.
        rollout_backend_class: Optional[str], class name of the rollout backend.

    Returns:
        Type[AsyncServerBase]: async server class.
    """
    if rollout_backend_class is None and rollout_backend_module is None:
        # If both are None, use the default backend class
        # Do not change the original import behavior
        # importlib.import_module and from ... import ... have subtle differences in ray

        if rollout_backend == "vllm":
            from siirl.engine.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer

            return AsyncvLLMServer
        elif rollout_backend == "sglang":
            from siirl.engine.rollout.sglang_rollout.async_sglang_server import AsyncSglangServer
            return AsyncSglangServer
        else:
            raise NotImplementedError(f"rollout backend {rollout_backend} is not supported")

    if rollout_backend_module is None or rollout_backend_class is None:
        raise ValueError("rollout_backend_module and rollout_backend_class must be both provided for customization")

    from siirl.utils.import_utils import load_extern_type

    return load_extern_type(rollout_backend_module, rollout_backend_class)
