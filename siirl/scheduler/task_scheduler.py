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

import collections
import re
from typing import List, Dict, Callable, Optional, Tuple, Set
from loguru import logger
from siirl.workers.dag.node import Node, NodeType
from siirl.workers.dag.task_graph import TaskGraph
from siirl.workers.dag.task_loader import discover_and_split_parallel_paths


def _parse_model_params_string(params_value: any) -> float:
    """
    Parses a model parameter value, which can be a number or a string with units (M/B/K).
    E.g.: "70B" -> 70 * 10^9, "500M" -> 500 * 10^6, "100K" -> 100 * 10^3.
    """
    if isinstance(params_value, (int, float)):
        return float(params_value)
    if isinstance(params_value, str):
        params_value_upper = params_value.upper()
        # Regex to extract the numerical part (integer or float)
        num_part_match = re.match(r"^\d+(\.\d+)?", params_value_upper)
        if not num_part_match:
            logger.warning(f"Could not parse numerical part from string '{params_value}'. Defaulting to 0.")
            return 0.0

        num = float(num_part_match.group(0))

        if params_value_upper.endswith("B"):
            return num * 1e9  # Billion
        elif params_value_upper.endswith("M"):
            return num * 1e6  # Million
        elif params_value_upper.endswith("K"):
            return num * 1e3  # Thousand
        else:
            # If no explicit unit, but the string can be converted to a float (e.g., "1000000")
            try:
                return float(params_value)  # Try to convert the whole string if no unit
            except ValueError:
                logger.warning(f"Unrecognized model parameter unit or format '{params_value}'. Defaulting to 0.")
                return 0.0
    logger.warning(f"Unknown model parameter type '{type(params_value)}' value '{params_value}'. Defaulting to 0.")
    return 0.0


def estimate_graph_model_params(task_graph: TaskGraph) -> float:
    """
    Estimates the 'size' of a task graph, typically based on the sum of model parameters
    for MODEL_TRAIN or MODEL_INFERENCE nodes.
    If no such nodes exist, it defaults to the number of nodes.
    If model nodes exist but have no 'model_params' in config, it defaults to 0.
    'model_params' can be a number, or a string like "70B" (70 billion) or "500M" (500 million).

    Args:
        task_graph (TaskGraph): The task graph to estimate.

    Returns:
        float: The estimated size of the graph.
    """
    total_params: float = 0.0
    if not task_graph or not task_graph.nodes:
        return 0.0

    has_model_nodes = False
    has_positive_params_in_model_nodes = False
    for node in task_graph.nodes.values():
        if node.node_type in [NodeType.MODEL_TRAIN, NodeType.MODEL_INFERENCE]:
            has_model_nodes = True
            # Get 'model_params' from node config, default to 0.0 if not present
            raw_params_value = node.config.get("model_params", 0.0)
            node_params = _parse_model_params_string(raw_params_value)  # Use helper to parse params

            if node_params > 0:
                has_positive_params_in_model_nodes = True
            total_params += node_params

    # If there are no model-related nodes, use the number of nodes as a proxy for size.
    if not has_model_nodes:
        return float(len(task_graph.nodes))
    # If there are model nodes but none have positive parameter counts,
    # this implies they are model-related but their sizes are unknown or zero.
    # Returning 0.0 signals that it's a model graph of unknown/zero parameter size.
    if has_model_nodes and not has_positive_params_in_model_nodes:
        return 0.0

    return total_params


class TaskScheduler:
    """
    Schedules tasks (represented as TaskGraphs) and assigns them to a distributed set of workers,
    each potentially having multiple GPUs. It aims to distribute tasks efficiently based on
    their size and cohesion (parts of the same task on the same node).
    """

    def __init__(self, num_physical_nodes: int, gpus_per_node: int):
        """
        Initializes the TaskScheduler.

        Args:
            num_physical_nodes (int): The total number of physical compute nodes (machines).
            gpus_per_node (int): The number of GPUs available on each physical node.

        Raises:
            ValueError: If num_physical_nodes or gpus_per_node is not positive.
        """
        if num_physical_nodes <= 0:
            raise ValueError("Number of physical nodes must be positive.")
        if gpus_per_node <= 0:
            raise ValueError("GPUs per node must be positive.")

        self.num_physical_nodes: int = num_physical_nodes
        self.gpus_per_node: int = gpus_per_node
        self.num_workers: int = num_physical_nodes * gpus_per_node  # Total available worker slots (GPUs)

        # State variables, reset for each scheduling call
        self.worker_to_graph_assignment: Dict[int, Optional[TaskGraph]] = {}  # Maps worker rank to assigned TaskGraph
        self.node_active_worker_count: Dict[int, int] = collections.defaultdict(int)  # Physical node index -> count of active workers
        self.node_free_gpus: Dict[int, List[int]] = collections.defaultdict(list)  # Physical node index -> list of free worker ranks (GPUs) on that node
        self._reset_scheduler_state()

    def _reset_scheduler_state(self):
        """Resets the internal state of the scheduler, typically before a new scheduling pass."""
        self.worker_to_graph_assignment = {r: None for r in range(self.num_workers)}
        self.node_active_worker_count = collections.defaultdict(int)
        self.node_free_gpus = collections.defaultdict(list)
        # Initialize free GPUs for each physical node
        for worker_rank in range(self.num_workers):
            physical_node_idx = worker_rank // self.gpus_per_node  # Determine which physical node this worker (GPU) belongs to
            self.node_free_gpus[physical_node_idx].append(worker_rank)

        for node_idx in self.node_free_gpus:
            self.node_free_gpus[node_idx].sort()  # Keep free GPU ranks sorted, useful for consistent tie-breaking

    def _get_original_graph_id(self, task_graph: TaskGraph) -> str:
        """
        Extracts the original graph ID from a potentially modified (e.g., split) TaskGraph ID.
        This is primarily used for logging or tracking purposes now, as inter-subgraph affinity
        is no longer a direct scheduling factor.

        Args:
            task_graph (TaskGraph): The TaskGraph whose original ID is to be determined.

        Returns:
            str: The original graph ID.
        """
        # Example naming from discover_and_split_parallel_paths: f"{base_id_for_final_naming}_split_{idx}"
        # This function assumes such a convention.
        parts = task_graph.graph_id.split("_split_")
        if len(parts) > 1:  # Covers cases like "origID_final_1", "origID_sub_1_split_2" etc.
            base_id_candidate = parts[0]
            return base_id_candidate
        return task_graph.graph_id  # Return full ID if pattern doesn't match

    def _apportion_workers_to_tasks(self, task_graphs_with_estimated_sizes: List[Tuple[TaskGraph, float]], num_total_workers_to_assign: int, apportion_strategy: str) -> Dict[str, int]:
        """
        Distributes a total number of workers among a list of tasks based on a chosen strategy.
        Ensures that if num_total_workers_to_assign >= num_tasks, each task gets at least one worker,
        with remaining workers distributed according to the strategy.

        Args:
            task_graphs_with_estimated_sizes (List[Tuple[TaskGraph, float]]): A list of tuples,
                each containing a TaskGraph and its estimated size. These are the tasks that *will* run.
            num_total_workers_to_assign (int): The total number of workers to distribute among these tasks.
            apportion_strategy (str): The strategy for apportioning workers ('even' or 'param_aware').

        Returns:
            Dict[str, int]: A dictionary mapping task_graph.graph_id to the number of workers assigned.
        """
        num_tasks_to_run = len(task_graphs_with_estimated_sizes)
        # Initialize apportionment: graph_id -> num_workers
        apportionment: Dict[str, int] = {tg.graph_id: 0 for tg, _ in task_graphs_with_estimated_sizes}

        if num_tasks_to_run == 0 or num_total_workers_to_assign == 0:
            return apportionment  # No tasks or no workers, nothing to apportion.

        workers_assigned_so_far = 0
        # Step 1: Assign one worker to each task that is set to run, if enough workers are available.
        # This ensures every task selected to run gets at least minimal resources.
        if num_total_workers_to_assign >= num_tasks_to_run:
            for graph, _ in task_graphs_with_estimated_sizes:
                apportionment[graph.graph_id] = 1
            workers_assigned_so_far = num_tasks_to_run

        # Calculate remaining workers to be distributed after the initial assignment (if any).
        remaining_workers_to_distribute = num_total_workers_to_assign - workers_assigned_so_far

        # Step 2: Distribute any remaining workers based on the chosen strategy.
        if remaining_workers_to_distribute > 0:
            if apportion_strategy == "even":
                # Distribute remaining workers as evenly as possible on top of the initial ones.
                if num_tasks_to_run == 0:  # Avoid division by zero if no tasks to run (already handled, but defensive)
                    return apportionment
                base_additional_workers = remaining_workers_to_distribute // num_tasks_to_run
                remainder_additional_workers = remaining_workers_to_distribute % num_tasks_to_run
                # Sort by graph ID for deterministic distribution of remainder, though original order is usually fine.
                sorted_graphs_for_remainder = sorted(task_graphs_with_estimated_sizes, key=lambda x: x[0].graph_id)
                for i, (graph, _) in enumerate(sorted_graphs_for_remainder):
                    apportionment[graph.graph_id] += base_additional_workers + (1 if i < remainder_additional_workers else 0)

            elif apportion_strategy == "param_aware":
                # Calculate total size for weighting, excluding tasks with zero or negative estimated size.
                total_size_for_weights = sum(size for _, size in task_graphs_with_estimated_sizes if size > 0)

                if total_size_for_weights > 0 and num_tasks_to_run > 0:  # Ensure num_tasks_to_run > 0 for modulo
                    # Greedily assign remaining workers one by one to tasks, prioritizing larger tasks.
                    # Tasks are sorted by size (desc) to pick recipients for extra workers.
                    # This ensures that larger tasks are favored when distributing the remainder.
                    sorted_tasks_by_size_desc = sorted(task_graphs_with_estimated_sizes, key=lambda x: x[1], reverse=True)

                    temp_rem_workers = remaining_workers_to_distribute
                    worker_idx_counter = 0
                    # Distribute remaining workers one by one, cycling through tasks sorted by size.
                    while temp_rem_workers > 0:
                        # Cycle through tasks (largest first) to give them additional workers.
                        task_to_give_worker_tuple = sorted_tasks_by_size_desc[worker_idx_counter % num_tasks_to_run]
                        apportionment[task_to_give_worker_tuple[0].graph_id] += 1
                        temp_rem_workers -= 1
                        worker_idx_counter += 1
                else:
                    # If all tasks have zero/negative size or no tasks, fall back to even distribution for the remainder.
                    if num_tasks_to_run == 0:
                        return apportionment  # Avoid division by zero
                    base_additional_workers = remaining_workers_to_distribute // num_tasks_to_run
                    remainder_additional_workers = remaining_workers_to_distribute % num_tasks_to_run
                    sorted_graphs_for_fallback = sorted(task_graphs_with_estimated_sizes, key=lambda x: x[0].graph_id)
                    for i, (graph, _) in enumerate(sorted_graphs_for_fallback):
                        apportionment[graph.graph_id] += base_additional_workers + (1 if i < remainder_additional_workers else 0)
            else:
                raise ValueError(f"Unknown apportionment strategy: {apportion_strategy}")

        # Sanity check: Ensure the total number of assigned workers matches the target.
        current_sum_workers = sum(apportionment.values())
        if current_sum_workers != num_total_workers_to_assign:
            # This indicates a potential logic error in apportionment.
            logger.error(f"Apportionment sum mismatch. Expected {num_total_workers_to_assign}, got {current_sum_workers}. Apportionment map: {apportionment}")
            return {}
            # Corrective action might be needed here in a production system.
        return apportionment

    def schedule_and_assign_tasks(
        self, original_task_graphs: List[TaskGraph], size_estimator: Callable[[TaskGraph], float] = estimate_graph_model_params, apportion_strategy: str = "param_aware", consider_node_cohesion: bool = True, consider_node_load: bool = True, consider_rank_preference: bool = True
    ) -> Dict[int, Optional[TaskGraph]]:
        """
        Schedules a list of original TaskGraphs by first splitting them into irreducible subgraphes,
        then assigning these subgraphes to workers (GPUs) across physical nodes.

        The method raises a ValueError if not all schedulable subgraphes can be assigned at least one worker.

        Args:
            original_task_graphs (List[TaskGraph]): A list of original TaskGraph objects to be scheduled.
            size_estimator (Callable[[TaskGraph], float]): A function that estimates the 'size' of a TaskGraph.
            apportion_strategy (str): Strategy for distributing workers among tasks.
            consider_node_cohesion (bool): If True, tries to schedule workers for the same
                                           irreducible subgraph onto the same physical compute node.
            consider_node_load (bool): If True (default), placement will prefer physical nodes with lower current load.
            consider_rank_preference (bool): If True (default), placement will prefer lower-ranked GPUs as a tie-breaker.


        Returns:
            Dict[int, Optional[TaskGraph]]: A dictionary mapping worker rank to its assigned TaskGraph.

        Raises:
            ValueError: If the number of schedulable subgraphes exceeds the number of available workers,
                        making it impossible to assign at least one worker to each.
        """
        self._reset_scheduler_state()  # Initialize scheduler state for a new run.

        if not original_task_graphs:
            logger.info("No original TaskGraphs provided for scheduling. All workers will be idle.")
            return self.worker_to_graph_assignment  # Return empty assignment if no tasks.

        # Step 1: Split original task graphs into irreducible subgraphes.
        # Irreducible subgraphes are the actual units of work that will be scheduled.
        all_irreducible_subgraphes: List[TaskGraph] = []
        for i, original_graph in enumerate(original_task_graphs):
            if not original_graph or not original_graph.nodes:
                logger.warning(f"Original graph at index {i} (ID: {original_graph.graph_id if original_graph else 'N/A'}) is empty. Skipping.")
                continue
            # discover_and_split_parallel_paths breaks down complex graphs.
            subgraphes = discover_and_split_parallel_paths(original_graph)
            all_irreducible_subgraphes.extend(subgraphes)

        if not all_irreducible_subgraphes:
            logger.info("No schedulable irreducible subgraphes were derived. All workers will be idle.")
            return self.worker_to_graph_assignment  # Return empty if splitting results in no subgraphes.

        # Step 2: Estimate sizes of irreducible subgraphes and sort them.
        # Sorting by size (descending) helps in prioritizing larger tasks if not all can run,
        # or in the 'param_aware' apportionment strategy.
        subgraphes_with_sizes_sorted: List[Tuple[TaskGraph, float]] = sorted(
            [(sg, size_estimator(sg)) for sg in all_irreducible_subgraphes],
            key=lambda x: x[1],  # Sort by estimated size.
            reverse=True,  # Largest tasks first.
        )

        num_schedulable_subgraphes = len(subgraphes_with_sizes_sorted)
        workers_per_task_map: Dict[str, int]  # Map: subgraph_id -> number of workers assigned to it.

        # Step 3: Determine how many workers each subgraph gets.
        # Crucially, if not all tasks can be assigned at least one worker, raise an error.
        if num_schedulable_subgraphes > self.num_workers:
            raise ValueError(f"Cannot assign all tasks. Number of schedulable subgraphes ({num_schedulable_subgraphes}) exceeds the total number of available workers ({self.num_workers}). Please provide more workers or reduce the number of tasks/subgraphes.")
        else:
            # All schedulable subgraphes will run.
            # Apportion all available workers (self.num_workers) among these subgraphes.
            tasks_to_run_with_sizes = subgraphes_with_sizes_sorted  # All of them are considered for running.
            workers_per_task_map = self._apportion_workers_to_tasks(
                tasks_to_run_with_sizes,  # All schedulable subgraphes.
                self.num_workers,  # Total workers to distribute among them.
                apportion_strategy,
            )

        # Filter to get only tasks that were actually assigned workers (should be all in this logic path).
        tasks_actually_running_with_sizes = [(tg, size) for tg, size in tasks_to_run_with_sizes if workers_per_task_map.get(tg.graph_id, 0) > 0]
        # Order tasks for placement: prioritize tasks needing more workers, then by size.
        # This can influence placement if certain nodes become full.
        tasks_to_place_ordered = sorted(
            tasks_actually_running_with_sizes,
            key=lambda x: (workers_per_task_map.get(x[0].graph_id, 0), x[1]),
            # Sort by num_workers_for_task then by size.
            reverse=True,  # Tasks needing more workers/larger tasks first.
        )

        # Step 4: Place each worker for each scheduled subgraph.
        for task_graph_to_place, _ in tasks_to_place_ordered:  # The estimated size is not directly used in this loop.
            subgraph_id = task_graph_to_place.graph_id
            num_workers_for_this_subgraph = workers_per_task_map.get(subgraph_id, 0)

            if num_workers_for_this_subgraph == 0:
                # This should not happen if the logic above correctly assigns workers.
                logger.warning(f"Subgraph {subgraph_id} was allocated 0 workers. Skipping placement.")
                continue

            # Keep track of workers assigned to *this specific subgraph instance* for cohesion calculation.
            workers_assigned_to_current_subgraph_instance: List[int] = []

            # Assign each of the required workers for the current subgraph.
            for worker_slot_index in range(num_workers_for_this_subgraph):
                best_worker_rank_for_slot: int = -1
                # Scoring tuple for placement: (cohesion_score, node_load_score, rank_preference_score)
                # Higher scores are better. node_load and rank_preference are negative (lower is better).
                best_placement_score: Tuple[float, float, float] = (float("-inf"), float("-inf"), float("-inf"))

                # Determine cohesion targets: physical nodes already running other workers for THIS specific subgraph instance.
                intra_task_cohesion_target_nodes: Set[int] = set()
                if consider_node_cohesion and workers_assigned_to_current_subgraph_instance:
                    for r_assigned_to_this_task in workers_assigned_to_current_subgraph_instance:
                        intra_task_cohesion_target_nodes.add(r_assigned_to_this_task // self.gpus_per_node)

                # Iterate over all physical compute nodes to find the best free GPU for the current slot.
                for physical_node_idx in range(self.num_physical_nodes):
                    if not self.node_free_gpus[physical_node_idx]:
                        continue  # No free GPUs on this physical node.

                    # Consider the first available (e.g., lowest rank) GPU on this node.
                    # Sorting of node_free_gpus[node_idx] in _reset_scheduler_state ensures this is deterministic.
                    potential_worker_rank = self.node_free_gpus[physical_node_idx][0]

                    # --- Calculate scores for this potential placement ---
                    # Cohesion score:
                    cohesion_score = 0.0
                    if consider_node_cohesion:
                        if not workers_assigned_to_current_subgraph_instance:
                            cohesion_score = 1.0  # Any node is fine for the first worker from cohesion perspective.
                        elif physical_node_idx in intra_task_cohesion_target_nodes:
                            cohesion_score = 1.0  # Placing with its peers.

                    # Node load score:
                    node_load_score_component = 0.0  # Default to 0 if not considering load.
                    if consider_node_load:
                        node_load_score_component = -float(self.node_active_worker_count[physical_node_idx])  # Negative: lower load is better.

                    # Rank preference score:
                    rank_score_component = 0.0  # Default to 0 if not considering rank preference.
                    if consider_rank_preference:
                        rank_score_component = -float(potential_worker_rank)  # Negative: lower rank is better (converted to float for type consistency in tuple).

                    current_placement_score: Tuple[float, float, float] = (cohesion_score, node_load_score_component, rank_score_component)

                    # If current placement is better than the best found so far, update.
                    if current_placement_score > best_placement_score:
                        best_placement_score = current_placement_score
                        best_worker_rank_for_slot = potential_worker_rank

                # Assign the subgraph to the best found worker slot.
                if best_worker_rank_for_slot != -1:
                    chosen_physical_node_idx = best_worker_rank_for_slot // self.gpus_per_node
                    self.worker_to_graph_assignment[best_worker_rank_for_slot] = task_graph_to_place
                    self.node_active_worker_count[chosen_physical_node_idx] += 1  # Increment active worker count for the chosen node.
                    self.node_free_gpus[chosen_physical_node_idx].remove(best_worker_rank_for_slot)  # Mark GPU as used.
                    workers_assigned_to_current_subgraph_instance.append(best_worker_rank_for_slot)
                else:
                    # This error implies a logic flaw if workers were apportioned but cannot be placed.
                    original_id_for_logging = self._get_original_graph_id(task_graph_to_place)  # For better logging.
                    logger.error(
                        f"Could not find any free worker to place for subgraph {subgraph_id} (Original: {original_id_for_logging}). "
                        f"Worker slot {worker_slot_index + 1}/{num_workers_for_this_subgraph}. "
                        f"Already placed for this subgraph: {workers_assigned_to_current_subgraph_instance}. "
                        f"Total workers assigned so far: {sum(1 for w in self.worker_to_graph_assignment.values() if w is not None)}/{self.num_workers}. "
                        f"Investigate scheduler logic or available resources."
                    )
                    break  # Stop trying to place workers for this subgraph if one fails critically.

        # Final check: Ensure all workers are utilized if there were tasks.
        final_assigned_worker_count = sum(1 for w_val in self.worker_to_graph_assignment.values() if w_val is not None)
        if final_assigned_worker_count != self.num_workers and len(all_irreducible_subgraphes) > 0:
            logger.warning(f"Post-scheduling, {self.num_workers - final_assigned_worker_count} workers are unexpectedly idle despite having {len(all_irreducible_subgraphes)} schedulable subgraphes. Total workers assigned: {final_assigned_worker_count}/{self.num_workers}.")

        return self.worker_to_graph_assignment

    def get_unique_assigned_task_graphs(self) -> Dict[str, TaskGraph]:
        """
        Returns a list of unique TaskGraph objects that have been assigned to workers
        as a result of the last scheduling pass. This list contains the irreducible
        subgraphs that were actually scheduled.

        Returns:
            List[TaskGraph]: A list of unique TaskGraph objects. Returns an empty list
                             if no tasks were scheduled or if the scheduler hasn't run.
        """
        if not self.worker_to_graph_assignment:
            return {}

        unique_graphs_map: Dict[str, TaskGraph] = {}
        for task_graph in self.worker_to_graph_assignment.values():
            if task_graph:  # Filter out None values (idle workers)
                # The graph_id of subgraphs generated by discover_and_split_parallel_paths
                # is unique, making it a reliable key for identifying unique TaskGraph instances.
                unique_graphs_map[task_graph.graph_id] = task_graph

        return unique_graphs_map


def _format_ranks_for_logging(ranks: Optional[List[int]], detailed_rank_printing: bool, threshold: int = 10) -> str:
    """
    Formats a list of ranks for logging.
    If detailed_rank_printing is True, or if the number of ranks is below threshold,
    it prints all ranks. Otherwise, it prints a range and count.
    """
    if not ranks:
        return "N/A"
    # Ensure ranks are sorted for consistent output, especially for range.
    # Make a copy before sorting if the original list should not be modified,
    # though in this context, the lists are usually temporary.
    sorted_ranks = sorted(list(set(ranks)))  # Remove duplicates and sort

    if detailed_rank_printing or len(sorted_ranks) <= threshold:
        return str(sorted_ranks)
    else:
        if not sorted_ranks:  # Should not happen if ranks is not None and not empty
            return "[] (Empty after sort/unique)"
        return f"[{min(sorted_ranks)} ... {max(sorted_ranks)}] (Count: {len(sorted_ranks)})"


def log_schedule_assignments(
    assignments: Dict[int, Optional["TaskGraph"]],
    num_total_workers: int,
    detailed_rank_printing: bool = False,  # New parameter
    rank_print_threshold: int = 10,  # New parameter
) -> None:
    """
    Clearly logs the results of task scheduling and assignment using loguru,
    with an option for concise rank printing.

    Args:
        assignments (Dict[int, Optional[TaskGraph]]):
            A dictionary where keys are worker ranks (int) and values are
            the assigned TaskGraph object or None.
        num_total_workers (int):
            The total number of available workers in the system.
        detailed_rank_printing (bool): If True, prints all ranks.
                                     Otherwise, uses range for large lists.
        rank_print_threshold (int): The threshold above which ranks are printed as a range.
    """
    if not isinstance(assignments, dict):
        # Use logger.error for actual errors, info for informational messages.
        logger.error("Input for printing schedule assignments must be a dictionary.")
        return

    log_messages: List[str] = ["\n\n--- Task Schedule Assignment Results ---", f"Detailed Rank Printing: {detailed_rank_printing}, Threshold: {rank_print_threshold}"]  # Collect all log parts here

    # 1. Group workers by TaskGraph
    task_to_workers_map: Dict[str, List[int]] = collections.defaultdict(list)
    idle_workers: List[int] = []

    # Iterate up to num_total_workers to correctly identify all idle workers
    all_assigned_ranks = set()
    for worker_rank, assigned_graph in assignments.items():
        if worker_rank < num_total_workers:  # Ensure we only consider valid worker ranks
            all_assigned_ranks.add(worker_rank)
            if assigned_graph and hasattr(assigned_graph, "graph_id"):
                task_to_workers_map[assigned_graph.graph_id].append(worker_rank)
            # else: # This rank is assigned None or an invalid object, consider idle if not in a task
            #    pass # Handled by the loop below

    for r in range(num_total_workers):
        if r not in all_assigned_ranks or assignments.get(r) is None:  # Check if rank is truly idle
            is_assigned_to_task = False
            for _, assigned_ranks_for_task in task_to_workers_map.items():
                if r in assigned_ranks_for_task:
                    is_assigned_to_task = True
                    break
            if not is_assigned_to_task:
                idle_workers.append(r)

    for graph_id_key in task_to_workers_map:  # Sort ranks within each task's list
        task_to_workers_map[graph_id_key].sort()
    idle_workers.sort()  # Sort idle worker list

    # 2. Prepare summary information
    num_assigned_workers = 0
    for worker_list in task_to_workers_map.values():
        num_assigned_workers += len(worker_list)
    # Correctly calculate idle workers based on total workers and those assigned tasks
    num_idle_workers = num_total_workers - num_assigned_workers

    num_scheduled_tasks = len(task_to_workers_map)

    log_messages.append(f"Total Workers: {num_total_workers}")
    log_messages.append(f"Workers with Assigned Tasks: {num_assigned_workers}")
    log_messages.append(f"Idle Workers: {num_idle_workers} (Derived from total - assigned)")
    log_messages.append(f"Number of Scheduled TaskGraphs (Subgraphs): {num_scheduled_tasks}")

    # 3. Prepare detailed assignment for each TaskGraph
    if task_to_workers_map:
        log_messages.append("\nDetailed Assignments:")
        for graph_id, worker_ranks in sorted(task_to_workers_map.items()):
            ranks_str = _format_ranks_for_logging(worker_ranks, detailed_rank_printing, rank_print_threshold)
            log_messages.append(f"  TaskGraph (Subgraph ID): {graph_id}")
            log_messages.append(f"    Assigned Worker Count: {len(worker_ranks)}")
            log_messages.append(f"    Worker Ranks: {ranks_str}")
    else:
        log_messages.append("\nNo TaskGraphs were assigned to any workers.")

    # 4. Prepare idle workers information
    # Use the derived idle_workers list for more accuracy if assignments dict might be sparse
    actual_idle_worker_ranks = [r for r in range(num_total_workers) if all(r not in wr for wr in task_to_workers_map.values())]
    actual_idle_worker_ranks.sort()

    if actual_idle_worker_ranks:
        ranks_str = _format_ranks_for_logging(actual_idle_worker_ranks, detailed_rank_printing, rank_print_threshold)
        log_messages.append("\nIdle Worker Ranks:")
        log_messages.append(f"  Ranks: {ranks_str} (Count: {len(actual_idle_worker_ranks)})")
    elif num_total_workers > 0:  # No idle workers, and there are workers in the system
        log_messages.append("\nNo idle workers.")

    if num_total_workers == 0:
        log_messages.append("\nSystem has no workers.")

    log_messages.append("--- End of Assignment Results ---\n\n")

    # Log all messages as a single multi-line info block
    logger.debug("\n".join(log_messages))


# --- Example Usage ---
if __name__ == "__main__":
    # Setup for creating dummy TaskGraph objects for testing
    def create_dummy_graph(graph_id: str, num_nodes: int, model_params: any = 0.0, dependencies_map: Optional[Dict[int, List[int]]] = None) -> TaskGraph:  # model_params type changed to any
        """Creates a TaskGraph for testing."""
        dummy_graph = TaskGraph(graph_id=graph_id)
        nodes_to_add = []
        for i in range(num_nodes):
            node_type_val = NodeType.COMPUTE
            node_config_val = {}
            current_node_id = f"{graph_id}_n{i}"
            node_deps: List[str] = []

            if dependencies_map and i in dependencies_map:
                node_deps = [f"{graph_id}_n{dep_idx}" for dep_idx in dependencies_map[i]]

            # Put model_params directly into config, let estimate_graph_model_params handle parsing
            if model_params != 0.0 and i == 0:  # For simplicity, assign params to the first node
                node_type_val = NodeType.MODEL_TRAIN
                node_config_val = {"model_params": model_params}  # Use passed model_params directly

            nodes_to_add.append(Node(node_id=current_node_id, node_type=node_type_val, config=node_config_val, dependencies=node_deps))

        if nodes_to_add:
            dummy_graph.add_nodes(nodes_to_add)
            dummy_graph.build_adjacency_lists()  # Important for validation and splitting
            is_valid, msg = dummy_graph.validate_graph()
            if not is_valid:
                logger.warning(f"Created dummy graph {graph_id} is invalid: {msg}")
        return dummy_graph

    # --- Scheduler Configuration ---
    num_physical_compute_nodes = 2  # e.g., 2 machines
    gpus_per_compute_node = 4  # e.g., 4 GPUs per machine
    scheduler = TaskScheduler(num_physical_nodes=num_physical_compute_nodes, gpus_per_node=gpus_per_compute_node)
    # Total workers = 2 * 4 = 8

    logger.info(f"--- Initialized Scheduler: {scheduler.num_physical_nodes} Physical Nodes, {scheduler.gpus_per_node} GPUs/Node, Total Workers: {scheduler.num_workers} ---")

    # --- Scenario 1: Fewer original tasks than workers ---
    logger.info("\n--- Scenario 1: 3 Original Tasks (irreducible), 8 Workers ---")
    original_tasks_scen1 = [
        create_dummy_graph("Weather_Sys", num_nodes=3, model_params="600B"),
        create_dummy_graph("NLP_Sys", num_nodes=2, model_params="300M"),
        create_dummy_graph("Vision_Sys", num_nodes=1, model_params=100.0),
    ]
    configs_to_test_scene1 = [
        {"name": "Apportion:Even, Cohesion:Y, Load:Y, Rank:Y", "apportion": "even", "cohesion": True, "load": True, "rank": True},
        {"name": "Apportion:Param, Cohesion:Y, Load:Y, Rank:Y", "apportion": "param_aware", "cohesion": True, "load": True, "rank": True},
        {"name": "Apportion:Param, Cohesion:N, Load:Y, Rank:Y", "apportion": "param_aware", "cohesion": False, "load": True, "rank": True},
        {"name": "Apportion:Param, Cohesion:Y, Load:N, Rank:Y", "apportion": "param_aware", "cohesion": True, "load": False, "rank": True},
        {"name": "Apportion:Param, Cohesion:Y, Load:Y, Rank:N", "apportion": "param_aware", "cohesion": True, "load": True, "rank": False},
    ]
    for cfg in configs_to_test_scene1:
        logger.info(f"\n-- Config: {cfg['name']} --")
        try:
            assignments = scheduler.schedule_and_assign_tasks(original_tasks_scen1, apportion_strategy=cfg["apportion"], consider_node_cohesion=cfg["cohesion"], consider_node_load=cfg["load"], consider_rank_preference=cfg["rank"])
            workers_per_scheduled_task = collections.defaultdict(list)
            for worker_rank, graph_obj in assignments.items():
                if graph_obj:
                    workers_per_scheduled_task[graph_obj.graph_id].append(worker_rank)

            logger.info("  Workers per TaskGraph (Subgraph ID):")
            for task_id, worker_ranks in sorted(workers_per_scheduled_task.items()):
                logger.info(f"    {task_id}: {len(worker_ranks)} workers (Ranks: {sorted(worker_ranks)})")
            logger.info(f"  Physical Node active worker counts: {dict(scheduler.node_active_worker_count)}")
            logger.info(f"  Physical Node free GPUs: {{node_idx: gpu_ranks for node_idx, gpu_ranks in scheduler.node_free_gpus.items() if gpu_ranks}}")
        except ValueError as e:
            logger.error(f"  Error during scheduling: {e}")

    # --- Scenario 2: More original tasks than workers (tasks are simple/irreducible) ---
    # This scenario should now raise a ValueError.
    logger.info("\n--- Scenario 2: 10 Original Tasks (irreducible), 8 Workers ---")
    original_tasks_scene2 = [create_dummy_graph(f"T{i}_Job", num_nodes=2, model_params=f"{(10 - i) * 50}M") for i in range(10)]

    logger.info(f"\n-- Config: Apportion:Even, Cohesion:Y, Load:Y, Rank:Y (Expecting ValueError) --")
    try:
        assignments_scene2 = scheduler.schedule_and_assign_tasks(original_tasks_scene2, apportion_strategy="even", consider_node_cohesion=True, consider_node_load=True, consider_rank_preference=True)
        workers_per_scheduled_task_scen2 = collections.defaultdict(list)
        for worker_rank, graph_obj in assignments_scene2.items():
            if graph_obj:
                workers_per_scheduled_task_scen2[graph_obj.graph_id].append(worker_rank)

        logger.info("  Workers per TaskGraph (Subgraph ID):")  # Should not be reached
        for task_id, worker_ranks in sorted(workers_per_scheduled_task_scen2.items()):
            logger.info(f"    {task_id}: {len(worker_ranks)} workers (Ranks: {sorted(worker_ranks)})")
        logger.info(f"  Physical Node active worker counts: {dict(scheduler.node_active_worker_count)}")
    except ValueError as e:
        logger.info(f"  Successfully caught expected error: {e}")

    # --- Scenario 3: An original task that *can* be split ---
    logger.info("\n--- Scenario 3: 1 Original Task (splittable), 8 Workers ---")
    original_splittable_graph = TaskGraph(graph_id="ex1_reconverge_orig")
    original_splittable_graph.add_nodes(
        [
            Node(node_id="A", node_type=NodeType.DATA_LOAD, config={"model_params": "10M"}),
            Node(node_id="B", node_type=NodeType.COMPUTE, dependencies=["A"]),
            Node(node_id="A1", node_type=NodeType.DATA_LOAD, config={"model_params": "10M"}),
            Node(node_id="B1", node_type=NodeType.COMPUTE, dependencies=["A1"]),
            Node(node_id="C", node_type=NodeType.COMPUTE, dependencies=["B", "B1"], config={"model_params": "50B"}),
            Node(node_id="D_ex1", node_type=NodeType.COMPUTE, dependencies=["C"]),
            Node(node_id="E_ex1", node_type=NodeType.MODEL_TRAIN, dependencies=["D_ex1"], config={"model_params": "100B"}),
        ]
    )
    original_splittable_graph.build_adjacency_lists()
    is_valid, msg = original_splittable_graph.validate_graph()
    if not is_valid:
        logger.error(f"Splittable graph is invalid: {msg}")

    original_tasks_scene3 = [original_splittable_graph]

    logger.info(f"\n-- Config: Apportion:Param, Cohesion:Y, Load:Y, Rank:Y --")
    try:
        assignments_scene3 = scheduler.schedule_and_assign_tasks(original_tasks_scene3, apportion_strategy="param_aware", consider_node_cohesion=True, consider_node_load=True, consider_rank_preference=True)
        workers_per_scheduled_task_scene3 = collections.defaultdict(list)
        for worker_rank, graph_obj in assignments_scene3.items():
            if graph_obj:
                workers_per_scheduled_task_scene3[graph_obj.graph_id].append(worker_rank)

        logger.info("  Workers per TaskGraph (Subgraph ID - after splitting ex1_reconverge_orig):")
        for task_id, worker_ranks in sorted(workers_per_scheduled_task_scene3.items()):
            original_source_graph_for_task = TaskGraph(graph_id=task_id)
            original_source = scheduler._get_original_graph_id(original_source_graph_for_task)
            logger.info(f"    {task_id} (from {original_source}): {len(worker_ranks)} workers (Ranks: {sorted(worker_ranks)})")
        logger.info(f"  Physical Node active worker counts: {dict(scheduler.node_active_worker_count)}")
    except ValueError as e:
        logger.error(f"  Error during scheduling for splittable graph: {e}")

    # --- Scenario 4: No tasks ---
    logger.info("\n--- Scenario 4: 0 Original Tasks, 8 Workers ---")
    try:
        assignments_scene4 = scheduler.schedule_and_assign_tasks([])
        logger.info(f"  Assignments (should be empty): {{k:v.graph_id if v else None for k,v in assignments_scen4.items() if v}}")
        logger.info(f"  Physical Node active worker counts (should be all zeros): {dict(scheduler.node_active_worker_count)}")
    except ValueError as e:  # Should not happen for no tasks
        logger.error(f"  Error during scheduling for no tasks: {e}")
