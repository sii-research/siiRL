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
import os
from typing import Any, Dict, List
from uuid import uuid4

from siirl.execution.rollout_flow.multiturn.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from contextlib import contextmanager
import time
import torch
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("SIIRL_LOGGING_LEVEL", "WARN"))

@contextmanager
def _timer(name: str, timing_dict: dict):
    """A context manager to measure execution time of a code block."""
    # if self.enable_perf:
        # torch.cuda.synchronize()
    start_time = time.perf_counter()
    yield
    # if self.enable_perf:
    #     torch.cuda.synchronize()
    end_time = time.perf_counter()
    timing_dict[name] = timing_dict.get(name, 0)  + end_time - start_time


class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)
        self.prompt_length = config.rollout.prompt_length
        self.response_length = config.rollout.response_length

    async def run(self, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        )

        with _timer("generate_sequences", metrics):
            response_ids = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=2,
            metrics=metrics,
        )
        return output
