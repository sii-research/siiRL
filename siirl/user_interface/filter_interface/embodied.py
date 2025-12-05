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

from collections import Counter
from typing import Any, Dict, List, Tuple

import torch
from loguru import logger
from tensordict import TensorDict

from siirl.params import SiiRLArguments
from siirl.dag_worker.data_structures import NodeOutput
from siirl.data_coordinator import SampleInfo
from siirl.data_coordinator.protocol import select_idxs

def verify(
    data: TensorDict,
) -> Tuple[List[float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Calculates scores and enriches the batch with accuracy information.

    This function uses the 'complete' field from the batch as the ground truth
    for scores. It then writes the calculated scores ('acc') and a format
    correctness tensor back into the input `data` object for downstream use.

    Args:
        data: The TensorDict object containing batch data, including 'responses'
            and 'complete' tensors.

    Returns:
        A tuple containing:
        - scores_list: A Python list of float scores for each sample.
        - reward_metrics: A dictionary of aggregate reward metrics (e.g., mean score).
        - format_metrics: A dictionary of aggregate format metrics.
        - reward_format_metrics: A dictionary of reward metrics excluding format issues.
    """
    # --- 1. Access Tensors and Metadata ---
    responses = data["responses"]
    completes = data["complete"]
    device = responses.device
    batch_size = responses.size(0)

    # --- 2. Sanity Check ---
    assert completes.size(0) == batch_size, "Batch size mismatch between 'completes' and 'responses'."

    # --- 3. Create Score Tensors ---
    scores_tensor = completes.float()
    # Assume format is always correct for this verification step.
    format_tensor = torch.ones(batch_size, dtype=torch.float32, device=device)

    data["acc"] = scores_tensor
    data["format_correctness"] = format_tensor

    # --- 4. Calculate Aggregate Metrics ---
    mean_score = scores_tensor.mean().item()
    reward_metrics = {"all": mean_score}
    format_metrics = {"all": 1.0}  # Always 1.0 based on the assumption above
    reward_format_metrics = {"all": mean_score}

    return scores_tensor.tolist(), reward_metrics, format_metrics, reward_format_metrics


def _filter_batch(batch: TensorDict, n_samples: int, config: SiiRLArguments) -> TensorDict:
    """Filters a batch based on accuracy and truncation criteria.

    Filtering is performed at the prompt level. If any of the `n_samples`
    responses for a single prompt fails a check, all `n_samples` responses
    for that prompt are discarded.

    Args:
        batch: The TensorDict object to be filtered. Must contain 'acc' tensor.
        n_samples: The number of responses generated per prompt.
        config: Configuration object containing filter settings.

    Returns:
        A new, potentially smaller, TensorDict object containing the filtered data.
    """
    device = batch["responses"].device
    num_prompts = len(batch) // n_samples

    # Access embodied sampling config (similar to DAPO's filter_groups)
    embodied_sampling = config.algorithm.embodied_sampling
    filter_accuracy = embodied_sampling.filter_accuracy
    
    # --- 1. Accuracy Filtering ---
    if filter_accuracy:
        # Reshape flat accuracy tensor into (num_prompts, n_samples)
        acc_matrix = batch["acc"].reshape(num_prompts, n_samples)
        # Calculate mean accuracy for each prompt
        prompt_mean_acc = acc_matrix.mean(dim=-1)

        # Log accuracy distribution when performance monitoring is enabled
        if config.dag.enable_perf:
            counts = Counter(prompt_mean_acc.tolist())
            num_prompts_debug = len(prompt_mean_acc)

            log_lines = [f"Accuracy Distribution ({num_prompts_debug} prompts):"]
            for score, count in sorted(counts.items()):
                log_lines.append(f"  - Score {score:.2f}: {count} prompts")

            logger.info("\n".join(log_lines))

        # Create a boolean mask for prompts within the desired accuracy bounds
        accuracy_lower_bound = embodied_sampling.accuracy_lower_bound
        accuracy_upper_bound = embodied_sampling.accuracy_upper_bound
        acc_mask = (prompt_mean_acc >= accuracy_lower_bound) & (prompt_mean_acc <= accuracy_upper_bound)
    else:
        # If disabled, create a mask that keeps all prompts
        acc_mask = torch.ones(num_prompts, dtype=torch.bool, device=device)

    # --- 2. Truncation Filtering ---
    filter_truncated = embodied_sampling.filter_truncated
    if filter_truncated:
        # For Embodied AI: check finish_step instead of response length
        if "finish_step" in batch:
            finish_steps = batch["finish_step"].reshape(num_prompts, n_samples)
            # Reuse env.max_steps directly (no need to duplicate in embodied_sampling)
            max_steps = config.actor_rollout_ref.embodied.env.max_steps
            
            # A prompt is considered truncated if *any* of its samples reached max steps
            has_truncated = (finish_steps >= max_steps).any(dim=-1)
            
            # Log truncation statistics for monitoring
            truncated_count = int(has_truncated.sum().item())
            non_truncated_count = len(has_truncated) - truncated_count
            logger.info(
                f"Truncation Distribution ({len(has_truncated)} prompts):\n"
                f"  - Truncated    : {truncated_count}\n"
                f"  - Non-truncated: {non_truncated_count}"
            )
            
            # Create a mask to keep only the non-truncated prompts
            trunc_mask = ~has_truncated
        else:
            logger.warning("No 'finish_step' field found in batch. Skipping truncation filtering.")
            trunc_mask = torch.ones(num_prompts, dtype=torch.bool, device=device)
    else:
        # If disabled, create a mask that keeps all prompts
        trunc_mask = torch.ones(num_prompts, dtype=torch.bool, device=device)

    # --- 3. Combine Masks and Apply Filter ---
    # A prompt is kept only if it passes both accuracy and truncation checks
    combined_mask = acc_mask & trunc_mask

    # Expand the prompt-level mask to the sample-level to match the batch dimension
    final_mask = combined_mask.repeat_interleave(n_samples)

    # Use select_idxs instead of slice for boolean mask filtering
    filtered_batch = select_idxs(batch, final_mask)
    logger.info(f"Filtered batch size: {len(filtered_batch)} (from original: {len(batch)})")

    return filtered_batch


def _compute_embodied_verification_metrics(
    batch: TensorDict,
    config: SiiRLArguments,
) -> Dict[str, float]:
    """
    Compute Embodied AI-specific metrics during verification phase.
    
    Args:
        batch: The batch being verified
        config: Configuration arguments
    
    Returns:
        Dictionary of Embodied verification metrics
    """
    try:
        from siirl.utils.embodied.metrics import (
            compute_rollout_metrics,
        )
        
        metrics = {}
        
        # Prepare batch dict for metrics computation
        batch_dict = {
            'responses': batch.batch.get('responses'),
            'complete': batch.batch.get('complete'),
            'finish_step': batch.batch.get('finish_step'),
        }
        
        # Add optional fields
        if 'pixel_values' in batch.batch:
            batch_dict['pixel_values'] = batch.batch['pixel_values']
        if 'acc' in batch.batch:
            batch_dict['acc'] = batch.batch['acc']
        
        # Compute rollout metrics
        rollout_metrics = compute_rollout_metrics(batch_dict, config)
        for key, value in rollout_metrics.items():
            # Add verify_ prefix to distinguish from actor metrics
            metrics[f"verify_{key}"] = value
        
        return metrics
        
    except Exception as e:
        logger.debug(f"Failed to compute Embodied verification metrics: {e}")
        return {}


def embodied_local_rank_sampling(
    config: SiiRLArguments,
    batch: TensorDict,
    **kwargs: Any,
) -> NodeOutput:
    """Performs verification, metric collection, and optional filtering on a batch.

    This function orchestrates the post-generation processing pipeline for a batch
    of samples. It first verifies all samples, then filters them according to
    configuration, and finally attaches the calculated metrics to the resulting batch.

    Args:
        config: Global SiiRL configuration arguments.
        batch: The input TensorDict batch from the generation stage.
        node_config: Configuration specific to this execution node.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        A NodeOutput object containing the processed (and potentially filtered) batch.
    """
    # Step 1: Verify the entire batch to get scores and enrich it with an 'acc' tensor.
    _, reward_metrics, format_metrics, reward_format_metrics = verify(batch)

    # Step 2: Build metrics dictionary with only useful metrics
    sample_metrics = {}
    
    # Step 3: Compute useful Embodied AI-specific verification metrics (if enabled)
    enable_embodied_metrics = True  # Default
    if hasattr(config, 'actor_rollout_ref') and hasattr(config.actor_rollout_ref, 'embodied'):
        if config.actor_rollout_ref.embodied is not None:
            if hasattr(config.actor_rollout_ref.embodied, 'enable_vla_metrics'):
                enable_embodied_metrics = config.actor_rollout_ref.embodied.enable_vla_metrics
    
    if enable_embodied_metrics:
        embodied_verification_metrics = _compute_embodied_verification_metrics(
            batch=batch,
            config=config,
        )
        sample_metrics.update(embodied_verification_metrics)

    # Step 4: Conditionally filter the batch.
    # Use algorithm.embodied_sampling config (aligned with DAPO's filter_groups approach)
    embodied_sampling = config.algorithm.embodied_sampling
    if embodied_sampling.filter_accuracy or embodied_sampling.filter_truncated:
        n_samples = config.actor_rollout_ref.rollout.n
        processed_batch = _filter_batch(batch, n_samples, config)
    else:
        # If filtering is disabled, the processed batch is the original batch.
        processed_batch = batch

    # Step 5: Ensure all tensors are on CPU before data rebalance
    # This fixes device mismatch issues where some tensors (task_id, trial_id) are on CUDA
    # while others (responses, etc.) are on CPU
    if processed_batch is not None:
        for key, tensor in processed_batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.device.type != 'cpu':
                processed_batch[key] = tensor.cpu()
                logger.debug(f"Moved {key} from {tensor.device} to CPU for data rebalance")

    return NodeOutput(batch=processed_batch, metrics=sample_metrics)