# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from typing import Any, Dict, List, Callable

import psutil
import numpy as np
import torch
import pandas as pd
import ray
from scipy.stats import mode
import logging
import os
from pathlib import Path
from datetime import datetime
from siirl import DataProto
from tensordict import TensorDict
from functools import partial
import json


def _compute_response_info(batch: TensorDict) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch["responses"].shape[-1]

    prompt_mask = batch["attention_mask"][:, :-response_length]

    if "response_mask" not in batch:
        response_mask = batch["attention_mask"][:, -response_length:]
    else:
        response_mask = batch["response_mask"]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def compute_data_metric(data: TensorDict):
    """  
        Computes various metrics from a batch of data for PPO training.
        This function calculates metrics related to scores, rewards, advantages, returns, values,
        and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
        for each metric category.
        Args:
            batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
            use_critic: Whether to include critic-specific metrics. Defaults to True.

        Returns:
            A dictionary of metrics including:
                - critic/score/mean, max, min: Statistics about sequence scores
                - critic/rewards/mean, max, min: Statistics about sequence rewards
                - critic/advantages/mean, max, min: Statistics about advantages
                - critic/returns/mean, max, min: Statistics about returns
                - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
                - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
                - response_length/mean, max, min, clip_ratio: Statistics about response lengths
                - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
                - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = data["token_level_scores"].sum(-1)
    sequence_reward = data["token_level_rewards"].sum(-1)

    advantages = data["advantages"]
    returns = data["returns"]

    max_response_length = data["responses"].shape[-1]
    prompt_mask = data["attention_mask"][:, :-max_response_length].bool()
    response_mask = data["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(data)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)
    
    if "values" in data:
        values = data["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)
        
    correct_threshold = 0.5
    rewards_per_response = data["token_level_rewards"].sum(-1)
    correct_mask = rewards_per_response > correct_threshold
    response_lengths = response_info["response_length"]

    # add by siirl
    correct_response_length =  response_lengths[correct_mask]
    wrong_response_length = response_lengths[~correct_mask]
    
    
    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if "values" in data
            else {}
        ),
        # response length
        "response/length/mean": torch.mean(response_length).detach().item(),
        "response/length/max": torch.max(response_length).detach().item(),
        "response/length/min": torch.min(response_length).detach().item(),
        "response/clip_ratio/mean": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        "response/correct_length/max": torch.max(correct_response_length).detach().item() if correct_response_length.numel() != 0 else 0,
        "response/correct_length/min": torch.min(correct_response_length).detach().item() if correct_response_length.numel() != 0 else 0,
        "response/correct_length/mean": torch.mean(correct_response_length).detach().item() if correct_response_length.numel() != 0 else 0,
        "response/wrong_length/max": torch.max(wrong_response_length).detach().item() if wrong_response_length.numel() != 0 else 0,
        "response/wrong_length/min": torch.min(wrong_response_length).detach().item() if wrong_response_length.numel() != 0 else 0,
        "response/wrong_length/mean": torch.mean(wrong_response_length).detach().item() if wrong_response_length.numel() != 0 else 0,
        
        # prompt length
        "prompt/length/mean": torch.mean(prompt_length).detach().item(),
        "prompt/length/max": torch.max(prompt_length).detach().item(),
        "prompt/length/min": torch.min(prompt_length).detach().item(),
        "prompt/clip_ratio/mean": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),

        # system_info
        "perf/process_cpu_mem_used_gb" : psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    }
    # multi-turn conversation
    if "__num_turns__" in data:
        num_turns = data["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()
    return metrics


def compute_timing_metrics(batch: TensorDict, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A Tensordict object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: TensorDict, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def _calculate_bootstrap_metrics(group: pd.DataFrame, variable_name: str, subset_size: int, n_bootstrap: int = 1000) -> Dict[str, float]:
    """Performs fully vectorized bootstrap sampling to estimate statistics.

    This is the core computational engine. It avoids all Python loops by using
    NumPy's vectorized indexing and Scipy's vectorized mode calculation to
    efficiently compute metrics for thousands of bootstrap samples at once.

    Args:
        group: DataFrame containing the data for a single prompt, including the
               target variable column and potentially a 'pred' column.
        variable_name: The name of the column to perform bootstrap sampling on.
        subset_size: The size of each bootstrap sample (referred to as 'N').
        n_bootstrap: The number of bootstrap iterations to perform.

    Returns:
        A dictionary containing the calculated mean and standard deviation for
        best-of-N, worst-of-N, and majority-vote-of-N metrics.
    """
    metrics = {}
    variable_values = group[variable_name].to_numpy()

    # --- Step 1: Generate all random indices for all bootstrap samples at once.
    # This creates a 2D array of shape (n_bootstrap, subset_size), where each
    # row is a set of indices for one bootstrap sample.
    bootstrap_indices = np.random.choice(len(variable_values), size=(n_bootstrap, subset_size), replace=True)

    # --- Step 2: Gather all bootstrap data samples using advanced indexing.
    # This efficiently creates a 2D array of the actual data values for all samples.
    bootstrap_data = variable_values[bootstrap_indices]

    # --- Step 3: Vectorized calculation for best-of-N and worst-of-N.
    # np.max/min along axis=1 finds the best/worst value within each sample.
    # The result is a 1D array of shape (n_bootstrap,).
    max_values_per_sample = np.max(bootstrap_data, axis=1)
    min_values_per_sample = np.min(bootstrap_data, axis=1)

    # Finally, calculate the mean and std across all bootstrap results.
    metrics[f"best@{subset_size}/mean"] = np.mean(max_values_per_sample)
    metrics[f"best@{subset_size}/std"] = np.std(max_values_per_sample)
    metrics[f"worst@{subset_size}/mean"] = np.mean(min_values_per_sample)
    metrics[f"worst@{subset_size}/std"] = np.std(min_values_per_sample)

    # --- Step 4: Vectorized calculation for majority vote ('maj').
    if "pred" in group.columns:
        prediction_values = group["pred"].to_numpy()
        bootstrap_predictions = prediction_values[bootstrap_indices]

        # Find the mode (most frequent prediction) for each bootstrap sample.
        # `scipy.stats.mode` is vectorized and can operate along an axis.
        modes_per_sample = mode(bootstrap_predictions, axis=1, keepdims=True)[0]

        # To get the value associated with the majority vote, we find the *first*
        # occurrence of the mode in each sample, replicating the original logic.
        # `argmax` on the boolean mask provides the index of the first `True`.
        mask = bootstrap_predictions == modes_per_sample
        first_match_indices = np.argmax(mask, axis=1)

        # Use the derived indices to gather the final majority vote values.
        # This requires indexing the i-th row of `bootstrap_data` with the i-th index.
        majority_values = bootstrap_data[np.arange(n_bootstrap), first_match_indices]

        metrics[f"maj@{subset_size}/mean"] = np.mean(majority_values)
        metrics[f"maj@{subset_size}/std"] = np.std(majority_values)

    return metrics


@ray.remote
def _process_prompt_group_task(group: pd.DataFrame, numeric_variables: List[str], seed: int) -> pd.DataFrame:
    """A Ray remote task to process metrics for a single prompt group.

    This function serves as the parallel unit of work. It takes a DataFrame
    for one prompt, calculates all standard and bootstrapped metrics, and
    returns a tidy DataFrame of the results.

    Args:
        group: DataFrame containing all data for a single prompt.
        numeric_variables: A list of column names to calculate metrics for.
        seed: The random seed to ensure reproducible results for this task.

    Returns:
        A tidy DataFrame with columns ['data_source', 'prompt', 'var_name',
        'metric_name', 'value'], containing all calculated metrics for the group.
    """
    # Seed the random number generator for this specific worker.
    np.random.seed(seed)

    # Extract identifying information from the group.
    data_source = group["data_source"].iloc[0]
    prompt = group["prompt"].iloc[0]
    num_responses = len(group)

    # Store results in a list of dictionaries for efficient DataFrame creation.
    results = []
    for var_name in numeric_variables:
        base_info = {"data_source": data_source, "prompt": prompt, "var_name": var_name}

        # --- Calculate standard (non-bootstrapped) metrics ---
        results.append({**base_info, "metric_name": f"mean@{num_responses}", "value": group[var_name].mean()})
        
        if num_responses > 1:
            # 1. Re-added the original std@N metric for the user's logging block.
            #    NOTE: Averaging this metric across prompts is statistically incorrect.
            results.append({**base_info, "metric_name": f"std@{num_responses}", "value": group[var_name].std(ddof=1)})

            # 2. Kept the components for the correct pooled standard deviation calculation.
            #    These will be used for the function's actual return value.
            variance = group[var_name].var(ddof=1)
            df = num_responses - 1
            sum_sq_dev = variance * df
            results.append({**base_info, "metric_name": "internal_sum_sq_dev_for_pooled_std", "value": sum_sq_dev})
            results.append({**base_info, "metric_name": "internal_df_for_pooled_std", "value": df})

            # --- Calculate bootstrapped metrics for various sample sizes ---
            bootstrap_sizes = sorted(list(set([2**i for i in range(1, 10) if 2**i < num_responses] + [num_responses])))

            for size in bootstrap_sizes:
                bootstrap_results = _calculate_bootstrap_metrics(group, var_name, subset_size=size)
                for metric_name, value in bootstrap_results.items():
                    results.append({**base_info, "metric_name": metric_name, "value": value})

    return pd.DataFrame(results)


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]

def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], sample_turns: list[int], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)
    metric_dict = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
            for metric_name, metric_val in metric2val.items():
                if (
                    (var_name == core_var)
                    and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                    and (f"@{n_max}" in metric_name)
                ):
                    metric_sec = "val-core"
                else:
                    metric_sec = "val-aux"
                pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                metric_dict[pfx] = metric_val

    if len(sample_turns) > 0:
        sample_turns = np.concatenate(sample_turns)
        metric_dict["val-aux/num_turns/min"] = sample_turns.min()
        metric_dict["val-aux/num_turns/max"] = sample_turns.max()
        metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()
    
    
    # calculate test_score per data source
    data_source_rewards = defaultdict(list)
    for data_source, reward in zip(data_sources, infos_dict['reward']):
        data_source_rewards[data_source].append(reward)

    for source, rewards in data_source_rewards.items():
        if rewards:
            metric_dict[f"val/test_score/{source}"] = np.mean(rewards)
    return metric_dict


def aggregate_validation_metrics(data_sources: List[str], sample_inputs: List[str], infos_dict: Dict[str, List[Any]], seed: int = 42) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = aggregate_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # --- 1. Data Consolidation ---
    # Combine all input lists into a single, unified DataFrame.
    df = pd.DataFrame({"data_source": data_sources, "prompt": sample_inputs, **infos_dict})
    numeric_vars = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]

    # --- 2. Task Preparation ---
    # Split the DataFrame into a list of smaller DataFrames, one for each prompt group.
    prompt_groups = [group for _, group in df.groupby(["data_source", "prompt"])]

    # --- 3. Parallel Dispatch ---
    # Launch all processing tasks concurrently. `ray.remote` returns immediately
    # with a future (ObjectRef) for each task.
    futures = [_process_prompt_group_task.remote(group, numeric_vars, int(seed)) for group in prompt_groups]

    # --- 4. Result Collection ---
    # `ray.get` blocks until all tasks are complete and retrieves their results.
    processed_df_list = ray.get(futures)
    
    if not processed_df_list:
        return {}
    processed_df = pd.concat(processed_df_list)    

    # --- 6. Final Aggregation ---
    # Perform a single, efficient groupby to get the mean value of each metric
    # across all prompts within a data source.
    # Separate the standard metrics from the internal components for pooled std.
    is_std_component = processed_df["metric_name"].str.startswith("internal_")
    is_legacy_std = processed_df["metric_name"].str.startswith("std@")
    
    # Exclude internal components AND the legacy std@N metric from the regular aggregation.
    regular_metrics_df = processed_df[~is_std_component & ~is_legacy_std]
    std_components_df = processed_df[is_std_component]

    # Aggregate regular metrics by taking the mean across all prompts.
    final_agg_df = regular_metrics_df.groupby(["data_source", "var_name", "metric_name"])["value"].mean().reset_index()

    final_df = final_agg_df
    # Calculate the pooled standard deviation correctly.
    if not std_components_df.empty:
        summed_components = std_components_df.groupby(["data_source", "var_name", "metric_name"])["value"].sum().unstack()
        total_df = summed_components["internal_df_for_pooled_std"]
        pooled_variance = summed_components["internal_sum_sq_dev_for_pooled_std"].divide(total_df).fillna(0)
        pooled_std = np.sqrt(pooled_variance)
        
        pooled_std_df = pooled_std.reset_index(name="value")
        pooled_std_df["metric_name"] = "pooled_std"
        
        final_df = pd.concat([final_agg_df, pooled_std_df], ignore_index=True)

    # --- 7. Output Formatting ---
    # Convert the flattened Series from the groupby into the required nested dict.
    output_dict = defaultdict(lambda: defaultdict(dict))
    for _, row in final_df.iterrows():
        output_dict[row['data_source']][row['var_name']][row['metric_name']] = row['value']

    return output_dict
