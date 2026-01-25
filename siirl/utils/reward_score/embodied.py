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

import re
from typing import Any, Dict, List, Tuple
from tensordict import TensorDict
# Handle different tensordict versions - NonTensorData location varies
try:
    from tensordict import NonTensorData
except ImportError:
    from tensordict.tensorclass import NonTensorData
import numpy as np
import torch
import torch.distributed as dist
from scipy import special
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def _tensor_to_str_list(tensor: torch.Tensor) -> List[str]:
    """Helper function to decode a byte tensor into a list of strings."""
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    byte_array = tensor.cpu().numpy()
    return [bytes(x).decode("utf-8", errors="ignore").rstrip("\0") for x in byte_array]


def _extract_task_name(task_file_name: str) -> str:
    """Helper function to parse the base task name from a trial file name."""
    match = re.match(r"(libero_\w+_task_\d+)_trial_\d+", task_file_name)
    return match.group(1) if match else task_file_name


def _compute_cluster_centers(embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
    """Compute cluster centers using DBSCAN clustering."""
    if len(embeddings) == 0:
        return np.array([])
    
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_embeddings)
    
    cluster_centers = []
    unique_labels = set(clustering.labels_) - {-1}  # Exclude noise label
    
    for label in unique_labels:
        cluster_points = scaled_embeddings[clustering.labels_ == label]
        center = scaler.inverse_transform(cluster_points.mean(axis=0, keepdims=True)).flatten()
        cluster_centers.append(center)
    
    # Fallback to mean if no clusters found
    if not cluster_centers:
        cluster_centers = [embeddings.mean(axis=0)]
    
    return np.array(cluster_centers)


def _extract_local_data(batch_data: TensorDict) -> Dict[str, Any]:
    """Extract local data from batch for reward computation."""
    batch_size = batch_data["responses"].size(0)
    completes = np.array(batch_data["complete"].tolist())
    finish_steps = batch_data["finish_step"].cpu().numpy()
    embeddings = batch_data["vjepa_embedding"].cpu().numpy()
    task_file_names = _tensor_to_str_list(batch_data["task_file_name"])
    task_names = np.array([_extract_task_name(name) for name in task_file_names])
    
    # Pre-filter zero embeddings
    zero_mask = np.all(embeddings == 0, axis=1)
    valid_mask = ~zero_mask
    
    return {
        "batch_size": batch_size,
        "embeddings": embeddings,
        "completes": completes,
        "finish_steps": finish_steps,
        "task_names": task_names,
        "valid_mask": valid_mask,
        "zero_mask": zero_mask,
    }


def _gather_all_data(
    local_data: Dict[str, Any],
    dp_size: int,
    tp_size: int,
    is_representative: bool
) -> Tuple[Dict[str, Any], List[int]]:
    """
    One-shot gather of all data from all DP ranks.
    
    All ranks in the world must participate in all_gather_object.
    Only representative ranks (tp_rank==0, last pp stage) send actual data.
    Non-representative ranks send None.
    
    Args:
        local_data: Local data dict
        dp_size: Number of DP ranks
        tp_size: Number of TP ranks (used to calculate representative rank indices)
        is_representative: Whether this rank is a representative rank
    
    Returns:
        Tuple of (global_data, batch_sizes_per_rank)
    """
    # Prepare data - only representative ranks send actual data
    if is_representative:
        send_data = {
            "batch_size": local_data["batch_size"],
            "embeddings": local_data["embeddings"].tolist(),
            "completes": local_data["completes"].tolist(),
            "task_names": local_data["task_names"].tolist(),
            "valid_mask": local_data["valid_mask"].tolist(),
        }
    else:
        send_data = None
    
    # All ranks must participate in all_gather_object
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    all_data = [None] * world_size
    dist.all_gather_object(all_data, send_data)
    
    # Filter to get only data from representative ranks
    # Representative ranks are: dp_rank * tp_size for each dp_rank (assuming tp_rank=0)
    representative_ranks = [i * tp_size for i in range(dp_size)]
    dp_data = [all_data[r] for r in representative_ranks if r < world_size and all_data[r] is not None]
    
    # Extract batch_sizes and merge data
    batch_sizes = [d["batch_size"] for d in dp_data]
    global_embeddings = np.concatenate([np.array(d["embeddings"]) for d in dp_data], axis=0)
    global_completes = np.concatenate([np.array(d["completes"]) for d in dp_data], axis=0)
    global_task_names = np.concatenate([np.array(d["task_names"]) for d in dp_data], axis=0)
    global_valid_mask = np.concatenate([np.array(d["valid_mask"]) for d in dp_data], axis=0)
    
    global_data = {
        "embeddings": global_embeddings,
        "completes": global_completes,
        "task_names": global_task_names,
        "valid_mask": global_valid_mask,
        "batch_size": len(global_embeddings),
    }
    
    return global_data, batch_sizes


def _compute_all_rewards(data: Dict[str, Any], logger) -> np.ndarray:
    """
    Compute rewards for all samples based on global data.
    
    Reward logic:
    - Success samples: reward = 1.0
    - Failed samples with valid embeddings: reward = sigmoid-shaped based on distance to success cluster centers
    - Invalid samples (zero embeddings): reward = 0.0
    """
    batch_size = data["batch_size"]
    embeddings = data["embeddings"]
    completes = data["completes"].astype(bool)
    task_names = data["task_names"]
    valid_mask = data["valid_mask"]
    
    final_rewards = np.zeros(batch_size, dtype=float)
    
    # Success + valid -> reward = 1.0
    success_mask = completes & valid_mask
    final_rewards[success_mask] = 1.0
    
    # Failed + valid -> reward shaping
    fail_mask = ~completes & valid_mask
    
    if not fail_mask.any():
        logger.info(f"[REWARD COMPUTE] No failed valid samples, skipping reward shaping")
        return final_rewards
    
    # Group by task and compute rewards
    unique_tasks = np.unique(task_names)
    
    for task in unique_tasks:
        task_mask = task_names == task
        task_success_mask = task_mask & success_mask
        task_fail_mask = task_mask & fail_mask
        
        success_count = task_success_mask.sum()
        fail_count = task_fail_mask.sum()
        
        if success_count == 0 or fail_count == 0:
            logger.info(f"[REWARD COMPUTE] Task '{task}': {success_count} success, {fail_count} failed - skipping")
            continue
        
        # Get embeddings
        success_emb = embeddings[task_success_mask]
        fail_emb = embeddings[task_fail_mask]
        fail_indices = np.where(task_fail_mask)[0]
        
        # Compute cluster centers from success embeddings
        cluster_centers = _compute_cluster_centers(success_emb)
        logger.info(f"[REWARD COMPUTE] Task '{task}': {success_count} success -> {len(cluster_centers)} clusters")
        
        # Compute distances from failed samples to nearest cluster center
        distance_matrix = cdist(fail_emb, cluster_centers, "euclidean")
        min_distances = distance_matrix.min(axis=1)
        
        # Normalize distances
        min_dist, max_dist = min_distances.min(), min_distances.max()
        dist_range = max_dist - min_dist
        
        if dist_range < 1e-6:
            normalized_dists = np.full_like(min_distances, 0.5)
        else:
            normalized_dists = (min_distances - min_dist) / dist_range
        
        # Sigmoid mapping: closer to success -> higher reward (max 0.6)
        sigmoid_steepness = 10.0
        sigmoid_offset = 0.5
        sigmoid_inputs = sigmoid_steepness * (sigmoid_offset - normalized_dists)
        reward_values = 0.6 * special.expit(sigmoid_inputs)
        
        final_rewards[fail_indices] = reward_values
        
        logger.info(f"[REWARD COMPUTE] Task '{task}': {fail_count} failed, "
                   f"dist range [{min_dist:.4f}, {max_dist:.4f}], "
                   f"rewards [{reward_values.min():.4f}, {reward_values.max():.4f}]")
    
    return final_rewards


def _build_results(
    global_rewards: np.ndarray,
    local_data: Dict[str, Any],
    dp_rank: int,
    batch_sizes: List[int]
) -> List[Dict[str, Any]]:
    """Build result list for local samples only."""
    local_batch_size = local_data["batch_size"]
    
    # Calculate slice based on pre-gathered batch_sizes (no extra gather needed)
    start_idx = sum(batch_sizes[:dp_rank])
    end_idx = start_idx + local_batch_size
    local_rewards = global_rewards[start_idx:end_idx]
    
    # Build results
    results = []
    for i in range(local_batch_size):
        results.append({
            "is_success": bool(local_data["completes"][i]),
            "task_name": local_data["task_names"][i],
            "format_correctness": 1.0,
            "is_zero_embedding": bool(local_data["zero_mask"][i]),
            "score": float(local_rewards[i]),
        })
    
    return results


def compute_embodied_reward(
    batch_data: TensorDict,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Computes rewards based on VJEPA embeddings and task completion status.
    
    Distributed-aware: uses one-shot all_gather to collect data from all DP ranks,
    then each rank computes the same global rewards and extracts its local slice.
    
    Args:
        batch_data: TensorDict containing batch information with parallelism info:
            - dp_size, dp_rank: Data Parallel info
            - tp_rank, tp_size: Tensor Parallel info  
            - pp_rank, pp_size: Pipeline Parallel info

    Returns:
        A list of dictionaries, each containing detailed score information.
    """
    from loguru import logger
    
    # === Step 1: Extract parallelism info from batch ===
    def get_nontensor_value(key, default):
        val = batch_data.get(key, None)
        if val is None:
            return default
        return val.data if isinstance(val, NonTensorData) else val
    
    dp_size = get_nontensor_value("dp_size", 1)
    dp_rank = get_nontensor_value("dp_rank", 0)
    tp_rank = get_nontensor_value("tp_rank", 0)
    tp_size = get_nontensor_value("tp_size", 1)
    pp_rank = get_nontensor_value("pp_rank", 0)
    pp_size = get_nontensor_value("pp_size", 1)
    
    logger.info(f"[REWARD COMPUTE] Parallelism: dp={dp_rank}/{dp_size}, tp={tp_rank}/{tp_size}, pp={pp_rank}/{pp_size}")
    
    # === Step 2: Extract local data ===
    local_data = _extract_local_data(batch_data)
    batch_size = local_data["batch_size"]
    
    logger.info(f"[REWARD COMPUTE] Local batch_size: {batch_size}, "
               f"success: {local_data['completes'].sum()}, "
               f"zero_emb: {local_data['zero_mask'].sum()}, "
               f"valid: {local_data['valid_mask'].sum()}")
    
    # === Step 3: Determine gather requirements ===
    # Only tp_rank==0 and last pp stage have actual data (representative ranks)
    is_representative = (tp_rank == 0) and (pp_rank == pp_size - 1)
    # All ranks must participate in gather if distributed, but only representative ranks have data
    need_distributed = dp_size > 1 and dist.is_initialized()
    
    # === Step 4: Gather all data (one-shot) or use local ===
    if need_distributed:
        logger.info(f"[REWARD COMPUTE] Rank (dp={dp_rank}, tp={tp_rank}, pp={pp_rank}): "
                   f"Participating in gather, is_representative={is_representative}")
        global_data, batch_sizes = _gather_all_data(local_data, dp_size, tp_size, is_representative)
        logger.info(f"[REWARD COMPUTE] dp_rank {dp_rank}: Global batch_size: {global_data['batch_size']}, "
                   f"batch_sizes_per_rank: {batch_sizes}")
    else:
        global_data = local_data
        batch_sizes = [batch_size]
    
    # === Step 5: Compute rewards on global data ===
    global_rewards = _compute_all_rewards(global_data, logger)
    
    # === Step 6: Build results for local samples ===
    if need_distributed:
        results = _build_results(global_rewards, local_data, dp_rank, batch_sizes)
    else:
        # Single rank or non-representative: results directly from global_rewards
        results = []
        for i in range(batch_size):
            results.append({
                "is_success": bool(local_data["completes"][i]),
                "task_name": local_data["task_names"][i],
                "format_correctness": 1.0,
                "is_zero_embedding": bool(local_data["zero_mask"][i]),
                "score": float(global_rewards[i]),
            })
    
    # === Step 7: Log final statistics ===
    local_rewards = np.array([r["score"] for r in results])
    num_success = (local_rewards == 1.0).sum()
    num_partial = ((local_rewards > 0) & (local_rewards < 1.0)).sum()
    num_failed = (local_rewards == 0).sum()
    
    logger.info(f"[REWARD COMPUTE] dp_rank {dp_rank}: Completed - "
               f"Avg: {local_rewards.mean():.4f}, Min: {local_rewards.min():.4f}, Max: {local_rewards.max():.4f}, "
               f"Success(1.0): {num_success}, Partial(0<r<1): {num_partial}, Failed(0): {num_failed}")
    logger.info(f"[REWARD COMPUTE] dp_rank {dp_rank}: rewards first 16: {local_rewards[:16].tolist()}")
    
    return results
