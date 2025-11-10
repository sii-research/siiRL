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
from typing import Any, Dict, List

import numpy as np
import torch
from scipy import special
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from siirl import DataProto


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


def compute_embodied_reward(
    batch_data: DataProto,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Computes rewards based on VJEPA embeddings and task completion status.

    This is the final, highly optimized version that implements a pre-filtering
    strategy. It identifies and excludes invalid samples (e.g., those with
    all-zero embeddings) *before* performing expensive computations like
    clustering and distance calculations, ensuring maximum efficiency.
    The results are logically and mathematically identical to the original implementation.

    Args:
        batch_data: The DataProto object containing all batch information.

    Returns:
        A list of dictionaries, each containing detailed score information.
    """
    # --- Step 1: Data Extraction and Global Pre-filtering ---
    from loguru import logger
    
    batch_size = batch_data.batch["responses"].size(0)
    completes = np.array(batch_data.batch["complete"].tolist())
    finish_steps = batch_data.batch["finish_step"].cpu().numpy()
    embeddings = batch_data.batch["vjepa_embedding"].cpu().numpy()
    task_file_names = _tensor_to_str_list(batch_data.batch["task_file_name"])

    # Pre-filtering: Identify all invalid samples (all-zero embeddings) upfront.
    zero_embedding_mask = np.all(embeddings == 0, axis=1)
    # Create an array of indices for only the valid samples.
    valid_indices = np.where(~zero_embedding_mask)[0]
    
    # Diagnostic logging
    num_zero = zero_embedding_mask.sum()
    num_success = completes.sum()
    num_valid = len(valid_indices)
    logger.info(f"[REWARD COMPUTE] Batch size: {batch_size}, Success: {num_success}, Zero embeddings: {num_zero}, Valid embeddings: {num_valid}")
    if num_zero == batch_size:
        logger.error(f"[REWARD COMPUTE] ALL EMBEDDINGS ARE ZERO! All rewards will be 0!")
    elif num_zero > 0:
        logger.warning(f"[REWARD COMPUTE] {num_zero}/{batch_size} embeddings are zero")

    # --- Step 2: Initialization ---
    # The final rewards array is initialized to zeros. Invalid samples will keep this score.
    final_rewards = np.zeros(batch_size, dtype=float)
    results: List[Dict[str, Any]] = [{} for _ in range(batch_size)]

    # Pre-populate all results with metadata. Scores for invalid samples will remain 0.
    task_names = [_extract_task_name(name) for name in task_file_names]
    for i in range(batch_size):
        results[i] = {
            "is_success": completes[i].item(),
            "task_name": task_names[i],
            "format_correctness": 1.0,
            "is_zero_embedding": zero_embedding_mask[i].item(),
            "score": 0.0,  # Default score is 0
        }

    # --- Step 3: Grouping and Reward Shaping on VALID samples only ---
    # Group only the valid samples by their task name.
    task_to_valid_indices = {}
    for idx in valid_indices:
        task_name = task_names[idx]
        task_to_valid_indices.setdefault(task_name, []).append(idx)

    # Process each task group using only the valid indices.
    for task_name, indices in task_to_valid_indices.items():
        indices = np.array(indices)
        task_completes = completes[indices]

        success_mask = task_completes
        fail_mask = ~success_mask

        success_indices = indices[success_mask]
        fail_indices = indices[fail_mask]
        
        logger.info(f"[REWARD COMPUTE] Processing task '{task_name}': {len(success_indices)} success, {len(fail_indices)} failed")

        # Assign base scores only for the valid samples.
        final_rewards[success_indices] = 1.0
        # Failed valid samples default to 0, may be updated by reward shaping below.

        if len(success_indices) == 0 or len(fail_indices) == 0:
            logger.info(f"[REWARD COMPUTE] Skipping task '{task_name}': no success or no failed samples for reward shaping")
            continue

        # --- Expensive computations now run only on the smaller, valid subset ---
        succ_embeddings = embeddings[success_indices]
        fail_embeddings = embeddings[fail_indices]

        # a. Clustering successful embeddings
        scaler = StandardScaler()
        scaled_succ_embeddings = scaler.fit_transform(succ_embeddings)
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(scaled_succ_embeddings)

        cluster_centers = []
        unique_labels = set(clustering.labels_)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in unique_labels:
            cluster_points = scaled_succ_embeddings[clustering.labels_ == label]
            center = scaler.inverse_transform(cluster_points.mean(axis=0, keepdims=True)).flatten()
            cluster_centers.append(center)

        if not cluster_centers:
            cluster_centers = [succ_embeddings.mean(axis=0)]

        cluster_centers = np.array(cluster_centers)

        # b. Vectorized distance calculation
        distance_matrix = cdist(fail_embeddings, cluster_centers, "euclidean")
        min_distances = distance_matrix.min(axis=1)

        # c. Vectorized reward mapping
        max_dist, min_dist = min_distances.max(), min_distances.min()
        dist_range = max_dist - min_dist

        if dist_range < 1e-6:
            normalized_dists = np.full_like(min_distances, 0.5)
        else:
            normalized_dists = (min_distances - min_dist) / dist_range

        sigmoid_steepness = 10.0
        sigmoid_offset = 0.5
        sigmoid_inputs = sigmoid_steepness * (sigmoid_offset - normalized_dists)
        reward_values = 0.6 * special.expit(sigmoid_inputs)

        # Assign shaped rewards back to the final array at their original indices.
        final_rewards[fail_indices] = reward_values

        # Populate detailed debug info back into the main results list.
        for i, idx in enumerate(fail_indices):
            results[idx]["distance_to_success"] = min_distances[i]
            results[idx]["normalized_distance"] = normalized_dists[i]

    # --- Step 4. Final Population of Score ---
    # Populate the final calculated scores from the numpy array into the results list.
    for i in range(batch_size):
        results[i]["score"] = final_rewards[i]
    
    # Consolidated statistics log
    num_success = (final_rewards == 1.0).sum()
    num_partial = ((final_rewards > 0) & (final_rewards < 1.0)).sum()
    num_failed = (final_rewards == 0).sum()
    logger.info(
        f"[REWARD COMPUTE] Batch {batch_size} completed - "
        f"Avg: {final_rewards.mean():.4f}, "
        f"Success (reward=1.0): {num_success}, "
        f"Partial (0<reward<1.0): {num_partial}, "
        f"Failed (reward=0): {num_failed}"
    )
    
    # Detailed per-sample information (debug level)
    for i in range(min(10, batch_size)):
        dist_info = f", dist={results[i].get('normalized_distance', 'N/A'):.4f}" if 'normalized_distance' in results[i] else ""
        logger.debug(f"[REWARD COMPUTE] Sample {i}: complete={completes[i]}, reward={final_rewards[i]:.4f}{dist_info}, zero_emb={results[i]['is_zero_embedding']}")

    return results
