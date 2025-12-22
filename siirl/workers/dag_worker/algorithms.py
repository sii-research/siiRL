# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import torch
from loguru import logger

from siirl import DataProto
from siirl.scheduler.enums import AdvantageEstimator
from siirl.utils.model_utils.torch_functional import masked_mean
from siirl.workers.dag_worker import core_algos


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    # responses = data.batch["responses"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80
    # a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.
    
    This function extracts the portion of the attention mask that corresponds 
    to the model's response. Handles both 2D responses (NLP) and 3D responses (Embodied AI).

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens (always 2D).
    """
    responses = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]
    batch_size = responses.size(0)
    
    # Handle 3D responses (Embodied AI): (batch_size, traj_len, action_token_len)
    if responses.ndim == 3:
        traj_len = responses.size(1)
        action_token_len = responses.size(2)
        
        # Check if attention_mask is also 3D
        if attention_mask.ndim == 3:
            # attention_mask: (batch_size, traj_len, tot_pad_len)
            # Extract response part from last dimension: (batch_size, traj_len, action_token_len)
            response_mask = attention_mask[:, :, -action_token_len:]
            # Flatten to 2D: (batch_size, traj_len * action_token_len)
            response_mask = response_mask.reshape(batch_size, -1)
        else:
            # attention_mask is 2D: (batch_size, total_length)
            # Calculate flattened response_length and slice
            response_length = traj_len * action_token_len
            response_mask = attention_mask[:, -response_length:]
    # Handle 2D responses (NLP): (batch_size, response_length)
    elif responses.ndim == 2:
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]
    else:
        raise ValueError(f"Unexpected responses shape: {responses.shape}, ndim={responses.ndim}")
    
    return response_mask


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    weight_factor_in_cpgd="STD_weight",
    **kwargs,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, CPGD, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++, CPGD).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO.
        Defaults to True.
        weight_factor_in_cpgd (str, optional): whether to use the STD weight as GRPO or clip_filter_like_weight.
        choices: {STD_weight, clip_filter_like_weight, naive}

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        
        # For embodied scenarios, use finish_step-based mask
        # Check if this is embodied scenario (has finish_step)
        if "finish_step" in data.batch and data.batch["responses"].ndim == 3:
            # Embodied scenario: compute mask based on finish_step
            responses = data.batch["responses"]
            batch_size = responses.size(0)
            response_length = responses.size(1) * responses.size(2)  # traj_len * action_token_len
            
            # Get action_token_len from config or infer from responses shape
            action_token_len = responses.size(2)  # action token length
            finish_step = data.batch['finish_step'] * action_token_len
            
            steps = torch.arange(response_length, device=responses.device)
            steps_expanded = steps.unsqueeze(0).expand(batch_size, -1)
            grpo_calculation_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
            
            logger.info(f"[GRPO] Using finish_step-based mask for embodied scenario")
        else:
            # NLP scenario or no finish_step: use attention_mask-based response_mask
            grpo_calculation_mask = data.batch["response_mask"]
            logger.info(f"[GRPO] Using attention_mask-based response_mask for NLP scenario")
        
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        # Store the mask for consistent metrics calculation
        data.batch["response_mask"] = grpo_calculation_mask
        logger.debug(f"[GRPO] Stored response_mask in batch for consistent metrics")
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.CPGD:
        # TODO: test on more adv estimator type
        cpgd_calculation_mask = data.batch["response_mask"]
        # Call compute_cpgd_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_cpgd_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=cpgd_calculation_mask,
            index=data.non_tensor_batch["uid"],
            weight_factor_in_cpgd=weight_factor_in_cpgd,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GAE_MARFT:
        core_algos.compute_marft_gae_advantage_return(
            data,
            pre_agent_group_ids=kwargs["agent_group_ids"],
            gamma=gamma,
            lam=lam,
        )
    else:
        raise NotImplementedError
    return data
