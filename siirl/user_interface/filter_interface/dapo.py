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

from collections import defaultdict
from typing import Any, Dict

import numpy as np
from tensordict import TensorDict
from siirl.params import SiiRLArguments
from siirl.dag_worker.data_structures import NodeOutput
from siirl.data_coordinator import DataProto
from siirl.data_coordinator.sample import filter_tensordict

def dynamic_sampling(siirl_args: SiiRLArguments, batch: TensorDict, node_config: Dict[str, Any], **kwargs: Any) -> NodeOutput:
    """
    Performs dynamic sampling by filtering trajectory groups based on metric variance.

    Args:
        siirl_args (SiiRLArguments): The global training arguments from the configuration.
        batch (DataProto): The input data batch for this step, which must contain 'uid'
                           in `non_tensor_batch` and the specified metric for filtering.
        node_config (Dict[str, Any]): The configuration specific to this node (not used here).
        **kwargs (Any): Additional keyword arguments (not used here).

    Returns:
        NodeOutput: An output object containing the filtered batch and metrics about the filtering process.

    Raises:
        KeyError: If the specified metric for filtering cannot be found in the batch and
                  cannot be computed from available data.
    """
    filter_config = siirl_args.algorithm.filter_groups

    # If filtering is disabled in the main config, bypass the logic and return the original batch.
    if not filter_config.enable:
        return NodeOutput(batch=batch, metrics={"sampling/kept_trajectories_ratio": 1.0})

    metric_name = filter_config.metric
    initial_traj_count = len(batch) if batch is not None else 0

    # Ensure the filtering metric exists. If not, try to compute it on-the-fly,
    # mirroring the behavior of dapo_ray_trainer.py.
    if metric_name not in batch:
        if metric_name == "seq_final_reward" and "token_level_rewards" in batch.batch:
            # Calculate from token-level rewards if necessary.
            batch["seq_final_reward"] = batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
        elif metric_name == "seq_reward" and "token_level_scores" in batch:
            # Calculate from token-level scores if necessary.
            batch["seq_reward"] = batch["token_level_scores"].sum(dim=-1).cpu().numpy()
        else:
            # If the metric cannot be found or computed, it's a configuration error.
            raise KeyError(f"Metric '{metric_name}' for group filtering not found in batch and could not be computed. Available non-tensor keys: {list(batch.keys())}")

    # Group trajectories by UID and collect their corresponding metric values.
    prompt_uid_to_metric_vals = defaultdict(list)
    uids = batch["uid"]
    metric_values = batch[metric_name]

    for i in range(len(uids)):
        prompt_uid_to_metric_vals[uids[i]].append(metric_values[i])

    # Calculate the standard deviation of the metric for each group of trajectories.
    prompt_uid_to_metric_std = {prompt_uid: np.std(metric_vals) for prompt_uid, metric_vals in prompt_uid_to_metric_vals.items()}

    # Decide which prompts (UIDs) to keep. A group is kept if its metric values
    # show variance (std > 0) or if it's a single-sample group (which cannot have variance).
    kept_prompt_uids = {uid for uid, std in prompt_uid_to_metric_std.items() if std > 0 or len(prompt_uid_to_metric_vals[uid]) == 1}

    # Find the indices of all trajectories that belong to the kept groups.
    # This ensures that all trajectories for a kept UID are preserved together.
    if not kept_prompt_uids:
        kept_traj_indices = []
    else:
        kept_traj_indices = [idx for idx, traj_uid in enumerate(uids) if traj_uid in kept_prompt_uids]

    # Filter the original batch by slicing it with the collected indices.
    # The DataProto object natively supports this slicing operation.
    filtered_batch = filter_tensordict(batch, kept_traj_indices)

    # Calculate and return metrics about the filtering process for logging and analysis.
    final_traj_count = len(filtered_batch) if filtered_batch is not None else 0
    kept_ratio = final_traj_count / initial_traj_count if initial_traj_count > 0 else 1.0
    metrics = {"dapo_sampling/kept_trajectories_ratio": kept_ratio, "dapo_sampling/initial_trajectories": initial_traj_count, "dapo_sampling/final_trajectories": final_traj_count, "dapo_sampling/kept_groups": len(kept_prompt_uids), "dapo_sampling/total_groups": len(prompt_uid_to_metric_vals)}

    return NodeOutput(batch=filtered_batch, metrics=metrics)
