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

from siirl import DataProto
from siirl.utils.reward_score import _default_compute_score
from siirl.models.transformers.internvl import IMG_CONTEXT_TOKEN
import torch
import os
import multiprocessing as mp
from functools import partial
import multiprocessing.dummy as mp_dummy
from functools import partial

class ParallelRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.rank = int(os.environ.get("RANK", "0"))
        self.reward_fn_key = reward_fn_key
    def _process_single_item(self, data_item):
        prompt_length = data_item.batch["prompts"].shape[-1]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        response_ids = data_item.batch["responses"]
        item = {
            "valid_response_ids": response_ids[:valid_response_length].cpu(),
            "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
            "data_source": data_item.non_tensor_batch[self.reward_fn_key],
            "extra_info": data_item.non_tensor_batch.get("extra_info", None),
        }
        return item

    def _compute_score(self, item):
        response_str = self.tokenizer.decode(item["valid_response_ids"])
        return self.compute_score(
            data_source=item["data_source"],
            solution_str=response_str,
            ground_truth=item["ground_truth"],
            extra_info=item["extra_info"],
        )

    def verify(self, data):
        with mp_dummy.Pool(processes=mp.cpu_count() // 2) as pool:
            items = [self._process_single_item(data[i]) for i in range(len(data))]
            scores = pool.map(partial(self._compute_score), items)
            data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data[0].batch["prompts"].device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = True):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        scores = self.verify(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            reward_tensor[i, valid_response_length - 1] = scores[i]

            data_source = data_item.non_tensor_batch["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if self.rank == 0 and already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                prompt_str = self.tokenizer.decode(data_item.batch["prompts"][-valid_prompt_length:])
                response_str = self.tokenizer.decode(data_item.batch["responses"][:valid_response_length])
                print("[prompt]", prompt_str.replace(IMG_CONTEXT_TOKEN, ""))
                print("[response]", response_str)
                print("[ground_truth]", data_item.non_tensor_batch["reward_model"]["ground_truth"])
                print("[score]", scores[i])
        if return_dict:
            return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": {},
                }
        return reward_tensor
