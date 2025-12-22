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
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from siirl.workers.dag import NodeType, TaskGraph


class ProcessGroupManager:
    """
    Manages the creation and assignment of process groups for distributed training.

    This class analyzes the topology of task graphs assigned to different workers (ranks)
    to determine which ranks need to communicate. It then defines process groups
    based on these communication patterns and provides methods to query these
    configurations.

    Attributes:
        total_num_workers (int): The total number of workers.
        ranks_taskgraph_mapping (Dict[int, Optional['TaskGraph']]): A mapping
            from a worker's rank to its assigned TaskGraph.
        relevant_node_types (Set[NodeType]): The set of node types to consider
            when forming process groups.
        process_group_spec (Dict[str, List[int]]): A mapping from a generated
            process group name to the list of ranks it contains.
        node_process_group_mapping (Dict[str, str]): A mapping from a node's ID
            to the name of the process group it belongs to.
    """

    def __init__(
        self,
        total_num_workers: int,
        ranks_taskgraph_mapping: Dict[int, Optional["TaskGraph"]],
        relevant_node_types_param: Optional[Set[NodeType]] = None,
    ):
        """Initializes the ProcessGroupManager.

        Args:
            total_num_workers: The total number of workers in the distributed setup.
            ranks_taskgraph_mapping: A mapping from worker ranks to their assigned TaskGraph.
            relevant_node_types_param: A set of NodeTypes to consider for group
                formation. If None, it defaults to MODEL_INFERENCE and MODEL_TRAIN.

        Raises:
            ValueError: If total_num_workers is not positive or if
                relevant_node_types_param has an invalid type.
        """
        if total_num_workers <= 0:
            raise ValueError("Total number of workers must be positive.")
        self.total_num_workers: int = total_num_workers
        self.ranks_taskgraph_mapping: Dict[int, Optional["TaskGraph"]] = dict(ranks_taskgraph_mapping)

        # --- Internal State Mappings ---
        # Maps a node ID to the sorted list of ranks that execute it.
        self.node_ranks_mapping: Dict[str, List[int]] = {}
        # Maps a process group name to its list of member ranks.
        self.process_group_spec: Dict[str, List[int]] = {}
        # Maps a node ID to its assigned process group name.
        self.node_process_group_mapping: Dict[str, str] = {}
        # Maps a node type (as a string) to the set of PGs associated with it.
        self.node_type_process_group_mapping: Dict[str, Set[str]] = collections.defaultdict(set)
        # Maps subgraph ID -> node type -> set of PG names.
        self.subgraph_node_type_pg_mapping: Dict[str, Dict[str, Set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))

        # Establish the set of node types that are relevant for process group creation.
        if relevant_node_types_param is None:
            self.relevant_node_types: Set[NodeType] = {
                NodeType.MODEL_INFERENCE,
                NodeType.MODEL_TRAIN,
            }
        else:
            if not isinstance(relevant_node_types_param, set) or not all(isinstance(nt, NodeType) for nt in relevant_node_types_param):
                raise ValueError("relevant_node_types_param must be a set of NodeType enums.")
            self.relevant_node_types: Set[NodeType] = relevant_node_types_param

        self._compute_group_configurations()

    def _clear_internal_mappings(self):
        """Resets all internal state dictionaries to an empty state."""
        self.node_ranks_mapping.clear()
        self.process_group_spec.clear()
        self.node_process_group_mapping.clear()
        self.node_type_process_group_mapping.clear()
        self.subgraph_node_type_pg_mapping.clear()

    def _collect_initial_topology_info(
        self,
    ) -> Tuple[Dict[str, Set[int]], Dict[str, "NodeType"], Dict[str, Set[str]]]:
        """
        Scans the rank-to-taskgraph mapping to build an initial understanding of the topology.

        It focuses only on nodes whose types are in `self.relevant_node_types`.

        Returns:
            A tuple containing:
            - graph_id_to_ranks: Mapping of a graph ID to the set of ranks running it.
            - node_id_to_type: Mapping of a relevant node ID to its NodeType.
            - graph_id_to_node_ids: Mapping of a graph ID to the set of relevant node IDs within it.
        """
        graph_id_to_ranks = collections.defaultdict(set)
        node_id_to_type: Dict[str, "NodeType"] = {}
        graph_id_to_node_ids: Dict[str, Set[str]] = collections.defaultdict(set)
        processed_graph_ids = set()

        for rank, tg_instance in self.ranks_taskgraph_mapping.items():
            if not tg_instance:
                continue

            gid = tg_instance.graph_id
            has_relevant_node = False

            # Process the structure of each unique graph only once.
            if gid not in processed_graph_ids:
                for node in tg_instance.nodes.values():
                    if node.node_type in self.relevant_node_types:
                        has_relevant_node = True
                        graph_id_to_node_ids[gid].add(node.node_id)
                        if node.node_id not in node_id_to_type:
                            node_id_to_type[node.node_id] = node.node_type
                if has_relevant_node:
                    processed_graph_ids.add(gid)
            # If graph was already processed, just check if it was deemed relevant.
            elif gid in graph_id_to_node_ids:
                has_relevant_node = True

            # If the graph contains any relevant nodes, associate the current rank with it.
            if has_relevant_node:
                graph_id_to_ranks[gid].add(rank)

        return graph_id_to_ranks, node_id_to_type, graph_id_to_node_ids

    def _aggregate_ranks_for_nodes(
        self,
        graph_id_to_ranks: Dict[str, Set[int]],
        graph_id_to_node_ids: Dict[str, Set[str]],
    ) -> Dict[str, Set[int]]:
        """
        Aggregates all ranks for each node based on the graph they belong to.

        Returns:
            A dictionary mapping each node ID to the complete set of ranks that execute it.
        """
        node_id_to_final_ranks = collections.defaultdict(set)
        for gid, nodes_in_graph in graph_id_to_node_ids.items():
            ranks_for_gid = graph_id_to_ranks.get(gid, set())
            if not ranks_for_gid:
                continue
            for node_id in nodes_in_graph:
                node_id_to_final_ranks[node_id].update(ranks_for_gid)
        return node_id_to_final_ranks

    def _populate_node_rank_mappings(self, node_id_to_final_ranks: Dict[str, Set[int]]):
        """Populates `self.node_ranks_mapping` with sorted lists of ranks for determinism."""
        for nid, ranks_set in node_id_to_final_ranks.items():
            self.node_ranks_mapping[nid] = sorted(list(ranks_set))

    def _define_process_groups(self) -> Dict[Tuple[int, ...], str]:
        """
        Defines process groups based on unique rank configurations found across all nodes.

        Returns:
            A mapping from a unique rank tuple to its generated process group name.
        """
        # Find all unique combinations of ranks that need to communicate.
        unique_rank_configs: Set[Tuple[int, ...]] = {tuple(ranks) for ranks in self.node_ranks_mapping.values()}

        # Assign a unique, deterministic name to each unique rank configuration.
        rank_config_to_group_name: Dict[Tuple[int, ...], str] = {}
        for i, rank_tuple in enumerate(sorted(list(unique_rank_configs))):
            group_name = f"process_group_{i + 1}"
            self.process_group_spec[group_name] = list(rank_tuple)
            rank_config_to_group_name[rank_tuple] = group_name
        return rank_config_to_group_name

    def _populate_final_node_and_type_assignments(
        self,
        rank_config_to_group_name: Dict[Tuple[int, ...], str],
        node_id_to_type: Dict[str, "NodeType"],
    ):
        """Populates the final mappings from node/type to process groups."""
        for nid, sorted_ranks_list in self.node_ranks_mapping.items():
            rank_tuple = tuple(sorted_ranks_list)
            group_name = rank_config_to_group_name.get(rank_tuple)
            if not group_name:
                continue

            self.node_process_group_mapping[nid] = group_name
            current_node_type = node_id_to_type.get(nid)
            if current_node_type:
                self.node_type_process_group_mapping[current_node_type.value].add(group_name)

    def _populate_subgraph_node_type_process_group_mapping(
        self,
        graph_id_to_node_ids: Dict[str, Set[str]],
        node_id_to_type: Dict[str, "NodeType"],
    ):
        """Populates the granular mapping of (subgraph, node_type) -> set of PGs."""
        for gid, nodes_in_graph in graph_id_to_node_ids.items():
            for node_id in nodes_in_graph:
                node_type = node_id_to_type.get(node_id)
                pg_name = self.node_process_group_mapping.get(node_id)

                if node_type and pg_name:
                    self.subgraph_node_type_pg_mapping[gid][node_type.value].add(pg_name)

    def _compute_group_configurations(self):
        """
        Orchestrates the step-by-step process of computing all process group configurations.
        """
        self._clear_internal_mappings()

        if not self.ranks_taskgraph_mapping:
            logger.warning("Ranks to TaskGraph mapping is empty. No process groups to compute.")
            return

        # Step 1: Discover the basic topology of graphs, nodes, and ranks.
        graph_id_to_ranks, node_id_to_type, graph_id_to_node_ids = self._collect_initial_topology_info()

        if not node_id_to_type:
            logger.warning("No nodes of a relevant type found. No process groups formed.")
            return

        # Step 2: Determine the full set of ranks for each node.
        node_id_to_final_ranks = self._aggregate_ranks_for_nodes(graph_id_to_ranks, graph_id_to_node_ids)
        self._populate_node_rank_mappings(node_id_to_final_ranks)

        if not self.node_ranks_mapping:
            return

        # Step 3: Define unique process groups from the rank configurations.
        rank_config_to_pg_name = self._define_process_groups()

        # Step 4: Create the final mappings for nodes and types.
        self._populate_final_node_and_type_assignments(rank_config_to_pg_name, node_id_to_type)
        self._populate_subgraph_node_type_process_group_mapping(graph_id_to_node_ids, node_id_to_type)

    # --- Public API Methods ---

    def get_group_spec(self, group_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves the specification (list of ranks) for a single process group."""
        ranks_list = self.process_group_spec.get(group_name)
        return {"ranks": ranks_list} if ranks_list is not None else None

    def get_all_specs(self) -> Dict[str, Dict[str, Any]]:
        """Retrieves all defined process group specifications."""
        return {name: {"ranks": ranks} for name, ranks in self.process_group_spec.items()}

    def get_node_assignment(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the rank and process group assignment for a specific node."""
        if node_id in self.node_process_group_mapping:
            return {
                "ranks": self.node_ranks_mapping[node_id],
                "process_group_name": self.node_process_group_mapping[node_id],
            }
        return None

    def get_process_groups_for_node_type(self, node_type_value: str) -> Set[str]:
        """Gets all process groups associated with a given node type globally."""
        # Return only for types that were configured as relevant during initialization.
        if any(rt.value == node_type_value for rt in self.relevant_node_types):
            return self.node_type_process_group_mapping.get(node_type_value, set())
        return set()

    def get_process_group_for_node_type_in_subgraph(self, graph_id: str, node_type_value: str) -> Set[str]:
        """Gets all process groups for a node type within a specific subgraph."""
        # Return only for types that were configured as relevant during initialization.
        if any(rt.value == node_type_value for rt in self.relevant_node_types):
            return self.subgraph_node_type_pg_mapping.get(graph_id, {}).get(node_type_value, set())
        return set()


# ==============================================================================
# Logging Utility Functions
# ==============================================================================


def _format_ranks_for_logging(ranks: Optional[List[int]], detailed_printing: bool, threshold: int = 10) -> str:
    """
    Formats a list of ranks for concise and readable logging.

    If detailed_printing is True or the list is short, it lists all ranks.
    Otherwise, it shows a compact range and count.

    Args:
        ranks: The list of integer ranks.
        detailed_printing: Flag to force printing all ranks.
        threshold: The number of ranks above which compact view is used.

    Returns:
        A formatted string representing the list of ranks.
    """
    if not ranks:
        return "N/A"
    if detailed_printing or len(ranks) <= threshold:
        return str(sorted(ranks))  # Sort for consistent output
    else:
        return f"[{min(ranks)}...{max(ranks)}] (Count: {len(ranks)})"


def _log_group_specs_report(pgm: ProcessGroupManager, detailed_printing: bool, threshold: int) -> List[str]:
    """Generates the log report for all process group specifications."""
    report_lines = []
    all_specs = pgm.get_all_specs()
    if not all_specs:
        report_lines.append("No process group specifications found.")
        return report_lines

    report_lines.append("All Process Group Specifications:")
    for group_name, spec in sorted(all_specs.items()):
        ranks_str = _format_ranks_for_logging(spec.get("ranks"), detailed_printing, threshold)
        report_lines.append(f"  - Group '{group_name}': Ranks {ranks_str}")
    return report_lines


def _log_node_assignments_report(
    pgm: ProcessGroupManager,
    nodes_to_query: Optional[List[str]],
    detailed_printing: bool,
    threshold: int,
) -> List[str]:
    """Generates the log report for node-to-process-group assignments."""
    report_lines = []
    all_mapped_nodes = sorted(pgm.node_process_group_mapping.keys())

    # Determine which nodes to generate the report for.
    if nodes_to_query is None:
        nodes_for_report = all_mapped_nodes
    else:
        nodes_for_report = nodes_to_query

    if not all_mapped_nodes:
        report_lines.append("No relevant node assignments found.")
        return report_lines

    report_lines.append("Node Assignments:")
    if nodes_to_query is None:
        report_lines.append(f"  (Logging all {len(all_mapped_nodes)} relevant node assignments)")

    for node_id in nodes_for_report:
        assignment = pgm.get_node_assignment(node_id)
        if assignment:
            ranks_str = _format_ranks_for_logging(assignment.get("ranks"), detailed_printing, threshold)
            report_lines.append(f"  - Node '{node_id}': Assigned to PG '{assignment['process_group_name']}', Ranks {ranks_str}")
        # Only report missing if it was specifically requested.
        elif node_id in (nodes_to_query or []):
            report_lines.append(f"  - Node '{node_id}': No assignment found (or not a relevant node).")

    return report_lines


def _log_global_type_mappings_report(pgm: ProcessGroupManager, node_types_to_query: Optional[List[NodeType]]) -> List[str]:
    """Generates the log report for global node type to process group mappings."""
    report_lines = []

    # Determine which node types to query.
    if node_types_to_query is None:
        types_for_report = sorted(pgm.relevant_node_types, key=lambda nt: nt.value)
    else:
        types_for_report = node_types_to_query

    if not pgm.relevant_node_types:
        report_lines.append("No node types were configured as relevant in the PGM.")
        return report_lines

    report_lines.append("Process Groups per Node Type (Global):")
    if node_types_to_query is None:
        report_lines.append("  (Logging for all configured relevant node types)")

    for node_type in types_for_report:
        pg_names = pgm.get_process_groups_for_node_type(node_type.value)
        if pg_names:
            report_lines.append(f"  - NodeType '{node_type.value}': Associated with PGs {sorted(list(pg_names))}")
        # Only report if the type was relevant but had no groups.
        elif node_type in pgm.relevant_node_types:
            report_lines.append(f"  - NodeType '{node_type.value}': No PGs found.")

    return report_lines


def _log_subgraph_mappings_report(
    pgm: ProcessGroupManager,
    subgraphs_to_query: Optional[List[str]],
    node_types_to_query: Optional[List[NodeType]],
) -> List[str]:
    """Generates the log report for subgraph-specific node type mappings."""
    report_lines = []

    # Determine which subgraphs and node types to query.
    all_mapped_subgraphs = sorted(pgm.subgraph_node_type_pg_mapping.keys())
    subgraphs_for_report = subgraphs_to_query or all_mapped_subgraphs
    types_for_report = node_types_to_query or sorted(pgm.relevant_node_types, key=lambda nt: nt.value)

    if not all_mapped_subgraphs:
        report_lines.append("No subgraph-specific mappings found.")
        return report_lines

    report_lines.append("Process Groups per NodeType within Subgraphs:")
    if subgraphs_to_query is None:
        report_lines.append(f"  (Logging for all {len(all_mapped_subgraphs)} available subgraphs)")

    for subgraph_id in subgraphs_for_report:
        if subgraph_id not in pgm.subgraph_node_type_pg_mapping:
            if subgraphs_to_query is not None:
                report_lines.append(f"  Subgraph ID: '{subgraph_id}' - No mappings found.")
            continue

        report_lines.append(f"  Subgraph ID: '{subgraph_id}'")
        found_any_pg = False
        for node_type in types_for_report:
            pg_names = pgm.get_process_group_for_node_type_in_subgraph(subgraph_id, node_type.value)
            if pg_names:
                report_lines.append(f"    - NodeType '{node_type.value}': Associated with PGs {sorted(list(pg_names))}")
                found_any_pg = True

        if not found_any_pg:
            report_lines.append(f"    No process groups found for any of the queried node types in this subgraph.")

    return report_lines


def log_process_group_manager_details(
    pgm: ProcessGroupManager, specific_nodes_to_query: Optional[List[str]] = None, specific_node_types_to_query: Optional[List[NodeType]] = None, specific_subgraphs_to_query: Optional[List[str]] = None, detailed_rank_printing: bool = False, rank_print_threshold: int = 16, log_level: str = "info"
):
    """
    Collects details from a ProcessGroupManager and logs them in a structured, aggregated message.

    This function provides a comprehensive snapshot of the PGM's state, including
    all defined process groups, node assignments, and type-based mappings,
    with options for controlling log verbosity.

    Args:
        pgm: The initialized ProcessGroupManager instance.
        specific_nodes_to_query: Optional list of node IDs to query for assignments.
        specific_node_types_to_query: Optional list of NodeTypes to query for PG associations.
        specific_subgraphs_to_query: Optional list of subgraph IDs to query.
        detailed_rank_printing: If True, prints all ranks; otherwise, uses a compact range for large lists.
        rank_print_threshold: The list size above which compact rank printing is used.
    """
    # NOTE: This function has been refactored into smaller helpers for clarity.
    # The final log output remains identical to the original version.

    header = [
        "\n\n--- Process Group Manager Details (Aggregated Log) ---",
        f"Detailed Rank Printing: {detailed_rank_printing}, Threshold: {rank_print_threshold}",
        "-" * 50,
    ]

    # Generate each section of the report using helper functions.
    specs_report = _log_group_specs_report(pgm, detailed_rank_printing, rank_print_threshold)
    nodes_report = _log_node_assignments_report(pgm, specific_nodes_to_query, detailed_rank_printing, rank_print_threshold)
    types_report = _log_global_type_mappings_report(pgm, specific_node_types_to_query)
    subgraphs_report = _log_subgraph_mappings_report(pgm, specific_subgraphs_to_query, specific_node_types_to_query)

    # Assemble the final log message.
    full_report = header + specs_report + ["-" * 50] + nodes_report + ["-" * 50] + types_report + ["-" * 50] + subgraphs_report + ["--- End of Process Group Manager Details (Aggregated Log) ---\n\n"]

    # Log the full report at the specified log level.
    valid_levels = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Choose from {valid_levels}.")
    logger.log(log_level.upper(), "\n".join(full_report))
