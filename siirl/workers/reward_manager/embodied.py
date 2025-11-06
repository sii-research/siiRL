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
        
        This method is called during validation to compute detailed metrics grouped by task.
        It follows the verl pattern where verification returns a 4-tuple of:
        (scores, reward_metrics, format_metrics, reward_format_metrics)
        
        Args:
            data: DataProto containing batch information with embodied task data
            
        Returns:
            tuple: (verifier_scores, reward_metrics, format_metrics, reward_format_metrics)
                - verifier_scores: List[float] - Score for each sample in the batch
                - reward_metrics: Dict[str, float] - Reward metrics aggregated by task and overall
                - format_metrics: Dict[str, float] - Format correctness metrics by task and overall
                - reward_format_metrics: Dict[str, float] - Combined metrics by task and overall
        
        Example return:
            verifier_scores = [0.8, 1.0, 0.3, 1.0, 0.0]
            reward_metrics = {
                'all': 0.62,
                'libero_spatial_task_1': 0.9,
                'libero_spatial_task_2': 0.5
            }
            format_metrics = {'all': 1.0, 'libero_spatial_task_1': 1.0, ...}
            reward_format_metrics = {'all': 0.62, ...}
        """
        logger.info(f"[DEBUG EmbodiedRewardManager.verify] ========== Validation 开始 ==========")
        
        batch_size = data.batch["responses"].shape[0]
        logger.info(f"[DEBUG EmbodiedRewardManager.verify] batch_size: {batch_size}")
        
        # --- Step 1: Compute rewards using the same logic as __call__ ---
        if self.compute_score is None:
            logger.error("[DEBUG EmbodiedRewardManager.verify] No compute_score function available!")
            # Return zero rewards as fallback
            verifier_scores = [0.0] * batch_size
            format_scores = [1.0] * batch_size
            scores_info = [
                {
                    "score": 0.0,
                    "format_correctness": 1.0,
                    "is_success": False,
                    "task_name": "unknown"
                }
                for _ in range(batch_size)
            ]
        else:
            # Delegate to compute_score function (e.g., compute_embodied_reward)
            logger.info(f"[DEBUG EmbodiedRewardManager.verify] 调用 compute_score")
            scores_info = self.compute_score(batch_data=data)
            verifier_scores = [info["score"] for info in scores_info]
            format_scores = [info.get("format_correctness", 1.0) for info in scores_info]
            logger.info(f"[DEBUG EmbodiedRewardManager.verify] verifier_scores前5个: {verifier_scores[:min(5, batch_size)]}")
            logger.info(f"[DEBUG EmbodiedRewardManager.verify] format_scores前5个: {format_scores[:min(5, batch_size)]}")
        
        # --- Step 2: Log examples for debugging (rank 0 only) ---
        if self.rank == 0 and self.print_count < self.num_examine:
            logger.info("--- EmbodiedRewardManager Validation Scoring ---")
            for i in range(min(batch_size, 2)):
                info = scores_info[i]
                logger.info(f"[Validation Sample {i}] Task: {info.get('task_name', 'N/A')}")
                logger.info(f"  - Success: {info.get('is_success')}")
                logger.info(f"  - Score: {info.get('score', 0.0):.4f}")
                if not info.get("is_success"):
                    dist = info.get("normalized_distance", "N/A")
                    if isinstance(dist, float):
                        logger.info(f"  - Normalized Distance: {dist:.4f}")
            self.print_count += 1
        
        # --- Step 3: Group by task name and aggregate metrics ---
        # This follows the verl pattern where metrics are broken down by task type
        logger.info(f"[DEBUG EmbodiedRewardManager.verify] 每个样本的详细信息:")
        for i in range(min(10, batch_size)):
            info = scores_info[i]
            logger.info(f"[DEBUG EmbodiedRewardManager.verify]   样本{i}: "
                       f"task={info.get('task_name', 'unknown')[:40]}, "
                       f"is_success={info.get('is_success')}, "
                       f"score={verifier_scores[i]:.4f}, "
                       f"format={format_scores[i]:.4f}, "
                       f"zero_emb={info.get('is_zero_embedding', False)}")
        
        task_groups = {}
        for i, info in enumerate(scores_info):
            task_name = info.get("task_name", "unknown")
            if task_name not in task_groups:
                task_groups[task_name] = {
                    "scores": [],
                    "formats": [],
                    "successes": []
                }
            task_groups[task_name]["scores"].append(verifier_scores[i])
            task_groups[task_name]["formats"].append(format_scores[i])
            task_groups[task_name]["successes"].append(info.get("is_success", False))
        
        # --- Step 4: Compute aggregated metrics ---
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
        
        # Overall metrics (required by verl)
        reward_metrics["all"] = float(np.mean(verifier_scores))
        format_metrics["all"] = float(np.mean(format_scores))
        reward_format_metrics["all"] = reward_metrics["all"]
        
        # Per-task metrics (for detailed analysis)
        for task_name, group_data in task_groups.items():
            task_scores = group_data["scores"]
            task_formats = group_data["formats"]
            task_successes = group_data["successes"]
            
            reward_metrics[task_name] = float(np.mean(task_scores))
            format_metrics[task_name] = float(np.mean(task_formats))
            reward_format_metrics[task_name] = reward_metrics[task_name]
            
            # Also compute success rate for each task (useful metric)
            success_rate_key = f"{task_name}_success_rate"
            reward_metrics[success_rate_key] = float(np.mean([float(s) for s in task_successes]))
        
        # Log summary statistics
        logger.info(f"[DEBUG EmbodiedRewardManager.verify] ========== Validation 完成 ==========")
        logger.info(f"[DEBUG EmbodiedRewardManager.verify] reward_metrics keys: {list(reward_metrics.keys())}")
        logger.info(f"[DEBUG EmbodiedRewardManager.verify] format_metrics keys: {list(format_metrics.keys())}")
        
        if self.rank == 0:
            logger.info(f"[EmbodiedRewardManager.verify] Batch summary:")
            logger.info(f"  - Overall mean reward: {reward_metrics['all']:.4f}")
            logger.info(f"  - Overall format correctness: {format_metrics['all']:.4f}")
            logger.info(f"  - Number of tasks: {len(task_groups)}")
            for task_name in sorted(task_groups.keys()):
                logger.info(
                    f"  - {task_name}: reward={reward_metrics[task_name]:.4f}, "
                    f"n={len(task_groups[task_name]['scores'])}"
                )
        
        logger.info(f"[DEBUG EmbodiedRewardManager.verify] 返回 4-tuple: (verifier_scores, reward_metrics, format_metrics, reward_format_metrics)")
        # Return the 4-tuple expected by validation_mixin.py
        return verifier_scores, reward_metrics, format_metrics, reward_format_metrics

