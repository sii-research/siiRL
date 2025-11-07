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
                        If False, returns format compatible with verl direct call
        
        Returns:
            If return_dict=True: {"reward_tensor": tensor, "reward_extra_info": dict}
            If return_dict=False: (reward_tensor_dict, reward_metrics) tuple
        """
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] ========== 开始 ==========")
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] return_dict: {return_dict}")
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] data类型: {type(data)}")
        
        batch_size = data.batch["responses"].shape[0]
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] batch_size: {batch_size}")
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] action_token_len: {self.action_token_len}")
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] reward_coef: {self.reward_coef}")

        # --- Step 1: Delegate the core reward calculation ---
        if self.compute_score is None:
            logger.error("[DEBUG EmbodiedRewardManager.__call__] No compute_score function available!")
            # Return zero rewards as fallback
            verifier_scores = [0.0] * batch_size
            format_scores = [1.0] * batch_size
            scores_info = [{"score": 0.0, "format_correctness": 1.0, "is_success": False} for _ in range(batch_size)]
        else:
            logger.info(f"[DEBUG EmbodiedRewardManager.__call__] 调用 compute_score 函数: {self.compute_score.__name__}")
            scores_info = self.compute_score(batch_data=data)
            logger.info(f"[DEBUG EmbodiedRewardManager.__call__] compute_score 返回了 {len(scores_info)} 个结果")
            verifier_scores = [info["score"] for info in scores_info]
            format_scores = [info.get("format_correctness", 1.0) for info in scores_info]
            logger.info(f"[DEBUG EmbodiedRewardManager.__call__] verifier_scores前5个: {verifier_scores[:min(5, batch_size)]}")
            logger.info(f"[DEBUG EmbodiedRewardManager.__call__] format_scores前5个: {format_scores[:min(5, batch_size)]}")

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
        
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] verifier_rewards shape: {verifier_rewards.shape}")
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] finish_step: {data.batch['finish_step']}")

        valid_response_length = data.batch["finish_step"] * self.action_token_len
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] valid_response_length: {valid_response_length}")

        for i in range(batch_size):
            last_step_idx = valid_response_length[i] - 1
            if last_step_idx >= 0:
                verifier_rewards[i, last_step_idx] = verifier_scores[i]
        
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] 填充后的verifier_rewards详细信息:")
        for i in range(min(5, batch_size)):
            non_zero_indices = torch.nonzero(verifier_rewards[i]).squeeze()
            task_info = scores_info[i]
            logger.info(f"[DEBUG EmbodiedRewardManager.__call__]   样本{i}: "
                       f"score={verifier_scores[i]:.4f}, "
                       f"is_success={task_info.get('is_success')}, "
                       f"finish_step={valid_response_length[i].item()//self.action_token_len}, "
                       f"last_step_idx={valid_response_length[i].item()-1}, "
                       f"非零位置={non_zero_indices.tolist() if non_zero_indices.numel() > 0 else '无'}")

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
        
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] ========== 计算完成 ==========")
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] reward_metrics: {reward_metrics}")
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] final_reward_tensor shape: {final_reward_tensor.shape}")
        logger.info(f"[DEBUG EmbodiedRewardManager.__call__] final_reward_tensor统计: mean={final_reward_tensor.mean():.4f}, max={final_reward_tensor.max():.4f}, min={final_reward_tensor.min():.4f}")

        # Return format based on return_dict flag
        if return_dict:
            # Format for compute_reward function (scheduler.reward.compute_reward)
            logger.info(f"[DEBUG EmbodiedRewardManager.__call__] 返回 dict 格式")
            return {
                "reward_tensor": reward_tensor_dict["all"],
                "reward_extra_info": reward_metrics
            }
        else:
            # Format for direct call (verl compatibility)
            logger.info(f"[DEBUG EmbodiedRewardManager.__call__] 返回 tuple 格式")
            return reward_tensor_dict, reward_metrics

    def verify(self, data: DataProto) -> Tuple[List[float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Verify and compute rewards for validation (verl-compatible interface).
        
        This method directly reads the 'complete' field from data.batch, following verl's
        RobRewardManager.verify() implementation exactly. No external compute_score is called.
        
        Args:
            data: DataProto containing batch information with embodied task data
            
        Returns:
            tuple: (verifier_scores, reward_metrics, format_metrics, reward_format_metrics)
                - verifier_scores: List[float] - Binary success (0/1) for each sample
                - reward_metrics: Dict[str, float] - Aggregated metrics
                - format_metrics: Dict[str, float] - Format correctness (always 1.0)
                - reward_format_metrics: Dict[str, float] - Same as reward_metrics
        """
        # Step 1: Read complete field directly from batch (verl pattern)
        completes = data.batch['complete'].tolist()
        batch_size = data.batch['responses'].size(0)
        assert len(completes) == batch_size
        
        # Convert boolean to float (0.0 or 1.0) - verl uses float(item)
        score = [float(item) for item in completes]
        
        # Step 2: Store to batch tensors (verl pattern, optimized)
        device = data.batch['responses'].device
        acc_tensor = torch.tensor(score, dtype=torch.float32, device=device)
        format_tensor = torch.ones(batch_size, dtype=torch.float32, device=device)
        
        data.batch['acc'] = acc_tensor
        data.batch['format_correctness'] = format_tensor
        
        # Step 3: Compute aggregated metrics (verl pattern)
        success_rate = acc_tensor.mean().item()
        
        reward_metrics = {'all': success_rate}
        format_metrics = {'all': 1.0}  # Always 1.0, no need to compute
        reward_format_metrics = {'all': success_rate}
        
        # Return the 4-tuple expected by validation_mixin.py (verl pattern)
        return score, reward_metrics, format_metrics, reward_format_metrics

