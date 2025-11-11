# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from transformers import PreTrainedTokenizer

from siirl import DataProto


class EmbodiedRewardManager:
    """
    Manages the reward calculation process for Embodied AI tasks.

    This class acts as an orchestrator. It receives the framework-specific
    `DataProto` object and delegates the complex reward computation to an
    injected `compute_score` function.
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        num_examine: int = 1,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        **reward_kwargs,
    ):
        """
        Initializes the reward manager.

        Args:
            tokenizer: The tokenizer, if needed for any text processing.
            num_examine: The number of reward examples to log for debugging.
            compute_score: The function to call for calculating reward scores.
                           Defaults to the optimized `compute_embodied_reward`.
            reward_fn_key: The key to identify the data source.
            **reward_kwargs: A dictionary for additional parameters like
                             `action_token_len` and `reward_coef`.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        
        # Import default compute_score if not provided
        if compute_score is None:
            try:
                from siirl.utils.reward_score.embodied import compute_embodied_reward
                self.compute_score = compute_embodied_reward
            except ImportError:
                logger.warning(
                    "Could not import compute_embodied_reward. "
                    "Please provide compute_score function or ensure embodied reward module exists."
                )
                self.compute_score = None
        else:
            self.compute_score = compute_score
            
        self.reward_fn_key = reward_fn_key
        self.rank = int(os.environ.get("RANK", "0"))
        self.print_count = 0

        # Extract specific parameters from kwargs with safe defaults.
        self.action_token_len = reward_kwargs.get("action_token_len", 7)
        self.reward_coef = reward_kwargs.get("reward_coef", 1.0)

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[Dict[str, Any], Tuple[Dict[str, torch.Tensor], Dict[str, float]]]:
        """
        Calculates and returns the reward tensors and metrics for a given data batch.
        
        Args:
            data: DataProto containing batch information
            return_dict: If True, returns format compatible with compute_reward function
                        If False, returns format compatible direct call
        
        Returns:
            If return_dict=True: {"reward_tensor": tensor, "reward_extra_info": dict}
            If return_dict=False: (reward_tensor_dict, reward_metrics) tuple
        """
        batch_size = data.batch["responses"].shape[0]

        # --- Step 1: Delegate the core reward calculation ---
        if self.compute_score is None:
            # Return zero rewards as fallback
            verifier_scores = [0.0] * batch_size
            format_scores = [1.0] * batch_size
            scores_info = [{"score": 0.0, "format_correctness": 1.0, "is_success": False} for _ in range(batch_size)]
        else:
            scores_info = self.compute_score(batch_data=data)
            verifier_scores = [info["score"] for info in scores_info]
            format_scores = [info.get("format_correctness", 1.0) for info in scores_info]

        # --- Step 3: Log debug examples (on rank 0 only) ---
        if self.rank == 0 and self.print_count < self.num_examine:
            logger.info("--- EmbodiedRewardManager Reward Calculation Example ---")
            for i in range(min(batch_size, 2)):
                info = scores_info[i]
                logger.info(f"Sample {i} | Task: {info.get('task_name', 'N/A')}")
                logger.info(f"  - Success: {info.get('is_success')}")
                if not info.get("is_success"):
                    dist = info.get("normalized_distance", "N/A")
                    if isinstance(dist, float):
                        logger.info(f"  - Normalized Distance: {dist:.4f}")
                    else:
                        logger.info(f"  - Normalized Distance: {dist}")
                logger.info(f"  -> Final Score: {info.get('score', 0.0):.4f}")
            self.print_count += 1

        # --- Step 4: Populate the reward tensor at the final timestep ---
        # The reward is applied as a terminal reward at the end of the action sequence.
        
        verifier_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        verifier_rewards = verifier_rewards.view(batch_size, -1)

        valid_response_length = data.batch["finish_step"] * self.action_token_len

        for i in range(batch_size):
            last_step_idx = valid_response_length[i] - 1
            if last_step_idx >= 0:
                verifier_rewards[i, last_step_idx] = verifier_scores[i]

        # --- Step 5: Aggregate final rewards and metrics ---
        reward_tensor_dict = {"gt_scores": verifier_rewards}
        reward_metrics = {}

        final_reward_tensor = torch.zeros_like(verifier_rewards)
        if self.reward_coef != 0:
            final_reward_tensor += self.reward_coef * reward_tensor_dict["gt_scores"]

            # Add all relevant metrics to the dictionary for logging.
            reward_metrics["verifier_mean"] = torch.tensor(verifier_scores).mean().item()
            reward_metrics["format_correctness_mean"] = torch.tensor(format_scores).mean().item()

        reward_tensor_dict["all"] = final_reward_tensor
        reward_metrics["reward_all"] = final_reward_tensor.sum(dim=-1).mean().item()
        

        # Return format based on return_dict flag
        if return_dict:
            # Format for compute_reward function (scheduler.reward.compute_reward)
            # Return per-sample format to match NaiveRewardManager/BatchRewardManager standard
            reward_extra_info = {
                "verifier_score": verifier_scores,      # Per-sample scores (already a list)
                "format_correctness": format_scores,    # Per-sample format correctness (already a list)
            }
            return {
                "reward_tensor": reward_tensor_dict["all"],
                "reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor_dict, reward_metrics

    def verify(self, data: DataProto) -> Tuple[List[float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Verify and compute rewards for validation.
        
        This method directly reads the 'complete' field from data.batch.
        
        Args:
            data: DataProto containing batch information with embodied task data
            
        Returns:
            tuple: (verifier_scores, reward_metrics, format_metrics, reward_format_metrics)
                - verifier_scores: List[float] - Binary success (0/1) for each sample
                - reward_metrics: Dict[str, float] - Aggregated metrics
                - format_metrics: Dict[str, float] - Format correctness (always 1.0)
                - reward_format_metrics: Dict[str, float] - Same as reward_metrics
        """
        # Step 1: Read complete field directly from batch
        completes = data.batch['complete'].tolist()
        batch_size = data.batch['responses'].size(0)
        assert len(completes) == batch_size
        
        # Convert boolean to float (0.0 or 1.0)
        score = [float(item) for item in completes]
        
        # Step 2: Store to batch tensors
        device = data.batch['responses'].device
        acc_tensor = torch.tensor(score, dtype=torch.float32, device=device)
        format_tensor = torch.ones(batch_size, dtype=torch.float32, device=device)
        
        data.batch['acc'] = acc_tensor
        data.batch['format_correctness'] = format_tensor
        
        # Step 3: Compute aggregated metrics
        success_rate = acc_tensor.mean().item()
        
        reward_metrics = {'all': success_rate}
        format_metrics = {'all': 1.0}  # Always 1.0, no need to compute
        reward_format_metrics = {'all': success_rate}
        
        # Return the 4-tuple expected by validation_mixin.py
        return score, reward_metrics, format_metrics, reward_format_metrics

