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

import os
from collections import defaultdict
from tensordict import TensorDict
import torch
from loguru import logger
from torch import distributed as dist

from siirl.utils.reward_score import default_compute_score


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.rank = int(os.environ.get("RANK", "0"))

    def __call__(self, data: TensorDict, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.keys():
            if return_dict:
                return {"reward_tensor": data["rm_scores"]}
            else:
                return data["rm_scores"]
        reward_tensor = torch.zeros_like(data["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i] 

            prompt_ids = data_item["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item["responses"]
            valid_response_length = data_item["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            from siirl.models.transformers.internvl import IMG_CONTEXT_TOKEN

            if IMG_CONTEXT_TOKEN in prompt_str:
                prompt_str = prompt_str.replace(IMG_CONTEXT_TOKEN, "")
            ground_truth = data["reward_model"][i]["ground_truth"]

            data_source = data[self.reward_fn_key][i]

            extra_info =  data["extra_info"][i] if "extra_info" in data else None
            
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if self.rank == 0 and already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                logger.info(f"rank:{dist.get_rank()}")
                logger.info(f"[prompt] {prompt_str}")
                logger.info(f"[response] {response_str}")
                logger.info(f"[ground_truth] {ground_truth}")
                if isinstance(score, dict):
                    for key, value in score.items():
                        logger.info(f"[{key}] {value}")
                else:
                    logger.info(f"[score] {score}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
