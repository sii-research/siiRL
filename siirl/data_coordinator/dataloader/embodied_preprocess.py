# Copyright 2025, Shanghai Innovation Institute.  All rights reserved.
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

from pathlib import Path
from typing import Tuple

import pandas as pd
from libero.libero import benchmark
from loguru import logger


def prepare_libero_train_valid_datasets(
    task_suite_name: str,
    num_trials_per_task: int,
    dataset_dir: str,
) -> Tuple[Path, Path]:
    """
    Generates identical training and validation dataset manifests for a LIBERO task suite.
    This manifest contains the necessary metadata for the VLA agent to initialize environments.

    The function queries the actual number of initial states for each task and generates
    trial_ids up to min(actual_initial_states, num_trials_per_task) to ensure all
    generated (task_id, trial_id) pairs are valid.

    Args:
        task_suite_name (str): The name of the task suite.
        num_trials_per_task (int): The maximum number of trials to include for each task.
                                   Actual trials may be less if a task has fewer initial states.
        dataset_dir (str): The directory where the manifest files will be saved.

    Examples:
        - Task with 50 initial states, num_trials_per_task=100 → generates trial_ids 0-49
        - Task with 150 initial states, num_trials_per_task=100 → generates trial_ids 0-99
        - Task with 150 initial states, num_trials_per_task=200 → generates trial_ids 0-149
    """
    # 1. --- Validate parameters and prepare output directory ---
    if num_trials_per_task < 1:
        raise ValueError("Number of trials per task must be at least 1")

    output_path = Path(dataset_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train.parquet"
    valid_file = output_path / "validate.parquet"

    # 2. --- Get task info from LIBERO benchmark ---
    try:
        task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
    except KeyError as err:
        raise ValueError(
            f"Task suite '{task_suite_name}' not found in benchmark."
        ) from err

    logger.info(f"Found {num_tasks_in_suite} tasks in '{task_suite_name}'.")
    logger.info(
        f"Requested maximum of {num_trials_per_task} trials per task."
    )

    # 3. --- Query actual initial state counts for each task ---
    logger.info("Querying initial state counts for each task...")
    task_initial_state_counts = []
    for task_id in range(num_tasks_in_suite):
        initial_states = task_suite.get_task_init_states(task_id)
        num_initial_states = len(initial_states)
        task_initial_state_counts.append(num_initial_states)
        logger.debug(
            f"Task {task_id}: {num_initial_states} initial states available"
        )

    # 4. --- Generate records with per-task trial_id limits ---
    logger.info("Generating dataset records with per-task limits...")
    all_records = []
    total_capped_tasks = 0
    
    for task_id in range(num_tasks_in_suite):
        actual_num_states = task_initial_state_counts[task_id]
        # Cap at actual available initial states
        max_trials_for_task = min(actual_num_states, num_trials_per_task)
        
        # Log if this task is being capped
        if max_trials_for_task < num_trials_per_task:
            logger.info(
                f"Task {task_id}: Capping at {max_trials_for_task} trials "
                f"(has {actual_num_states} initial states, requested {num_trials_per_task})"
            )
            total_capped_tasks += 1
        
        # Generate records for this task
        for trial_id in range(max_trials_for_task):
            all_records.append({
                "task_suite_name": task_suite_name,
                "task_id": task_id,
                "trial_id": trial_id,
                "prompt_id": f"{task_suite_name}_{task_id}_{trial_id}",
            })
    
    # Log summary statistics
    expected_records = num_tasks_in_suite * num_trials_per_task
    actual_records = len(all_records)
    logger.info(
        f"Generated {actual_records} records "
        f"(expected {expected_records} if all tasks had {num_trials_per_task} states)"
    )
    if total_capped_tasks > 0:
        logger.info(
            f"{total_capped_tasks} out of {num_tasks_in_suite} tasks were capped "
            f"due to insufficient initial states"
        )

    # 5. --- Save to both train and validation Parquet files ---
    try:
        if all_records:
            df = pd.DataFrame(all_records)
            df.to_parquet(train_file, index=False)
            df.to_parquet(valid_file, index=False)
            logger.success(
                f"✅ VLA task manifests successfully saved to '{output_path}'."
            )
        else:
            logger.warning("No records were generated for the VLA task manifest.")
    except ImportError:
        logger.error(
            "`pandas` and `pyarrow` are required. Please run: `pip install pandas pyarrow`"
        )
        raise
    except Exception as e:
        logger.error(f"Error saving Parquet file for VLA manifest: {e}")
        raise
    return train_file, valid_file

