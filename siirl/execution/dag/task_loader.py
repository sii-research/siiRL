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
import copy
from typing import Dict, List, Optional, Tuple, Set
import itertools

from loguru import logger
from siirl.execution.dag import Node, NodeType, TaskGraph


def generate_structural_signature(graph: TaskGraph) -> str:
    """
    Generates a canonical string signature for a TaskGraph based purely on its structure
    (nodes, their types, roles, and connections), ignoring the graph_id.

    Args:
        graph (TaskGraph): The TaskGraph object for which to generate the signature.

    Returns:
        str: A unique string representing the structural signature of the graph.
    """
    if not graph or not graph.nodes:
        # Handle empty or invalid graphs
        return f"empty_structure_original_id_ref:{graph.graph_id}"

    # Ensure adjacency lists are built for accurate dependency representation
    graph.build_adjacency_lists()

    # Sort node IDs to ensure consistent signature generation regardless of insertion order
    node_ids_sorted: List[str] = sorted(graph.nodes.keys())

    node_details_parts: List[str] = []
    # Collect details for each node, including its dependencies
    for nid in node_ids_sorted:
        node: Node = graph.nodes[nid]
        # Include sorted dependencies to make the node's structural role explicit
        sorted_deps: List[str] = sorted(list(node.dependencies))
        node_details_parts.append(f"n(id:{nid},t:{node.node_type.value},r:{node.node_role.value},d:[{','.join(sorted_deps)}])")

    # Explicitly list edges for robustness, based on the built adjacency list
    edge_list_parts: List[str] = []
    for parent_id in node_ids_sorted:
        children_ids_sorted: List[str] = sorted(graph.adj.get(parent_id, []))
        for child_id in children_ids_sorted:
            edge_list_parts.append(f"e({parent_id}->{child_id})")

    # Combine node and edge details into a single, canonical structural signature
    return f"struct_nodes:{';'.join(node_details_parts)}|struct_edges:{','.join(edge_list_parts)}"


def get_all_downstream_nodes_recursive(src_task_graph: TaskGraph, start_node_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
    """
    Recursively finds all downstream nodes reachable from a given start node in a TaskGraph.

    Args:
        src_task_graph (TaskGraph): The source TaskGraph to traverse.
        start_node_id (str): The ID of the starting node.
        visited (Optional[Set[str]]): A set of visited nodes to prevent infinite loops in cycles (though DAGs shouldn't have them).

    Returns:
        Set[str]: A set of all downstream node IDs, including the start node itself.
    """
    if visited is None:
        visited = set()

    # Base case: if node already visited or not in graph, return empty set
    if start_node_id in visited or start_node_id not in src_task_graph.nodes:
        return set()

    # Mark current node as visited and add to downstream set
    visited.add(start_node_id)
    downstream: Set[str] = {start_node_id}

    # Recursively find downstream nodes for each child
    for child_id in src_task_graph.adj.get(start_node_id, []):
        downstream.update(get_all_downstream_nodes_recursive(src_task_graph, child_id, visited.copy()))
    return downstream


def get_all_ancestors(graph: TaskGraph, node_id: str) -> Set[str]:
    """
    Finds all ancestor nodes of a given node in a TaskGraph.

    Args:
        graph (TaskGraph): The TaskGraph to traverse.
        node_id (str): The ID of the node for which to find ancestors.

    Returns:
        Set[str]: A set of all ancestor node IDs.
    """
    if node_id not in graph.nodes:
        return set()

    # Ensure reverse adjacency list is built for efficient ancestor traversal
    graph.build_adjacency_lists()

    ancestors: Set[str] = set()
    # Initialize queue with immediate parents
    queue: List[str] = list(graph.rev_adj.get(node_id, []))
    visited_ancestors: Set[str] = set(queue)
    ancestors.update(queue)

    head = 0
    # BFS traversal to find all ancestors
    while head < len(queue):
        current_node_id: str = queue[head]
        head += 1
        for parent_id in graph.rev_adj.get(current_node_id, []):
            if parent_id not in visited_ancestors:
                visited_ancestors.add(parent_id)
                ancestors.add(parent_id)
                queue.append(parent_id)
    return ancestors


def find_all_paths_dfs(src_task_graph: TaskGraph, current_node_id: str, end_node_id: str, current_path: List[str], all_paths: List[List[str]], visited_in_current_path: Set[str]):
    """
    Helper function for find_all_paths, uses DFS to find all paths between two nodes.

    Args:
        src_task_graph (TaskGraph): The source TaskGraph.
        current_node_id (str): The ID of the current node in the DFS traversal.
        end_node_id (str): The ID of the target end node.
        current_path (List[str]): The path built so far from the start node to current_node_id.
        all_paths (List[List[str]]): A list to accumulate all found paths.
        visited_in_current_path (Set[str]): Nodes visited in the current path to detect cycles (if graph were not a DAG).
    """
    current_path.append(current_node_id)
    visited_in_current_path.add(current_node_id)

    if current_node_id == end_node_id:
        # If end node reached, add a copy of the current path to all_paths
        all_paths.append(list(current_path))
    else:
        # Explore neighbors
        for neighbor_id in sorted(src_task_graph.adj.get(current_node_id, [])):
            if neighbor_id not in visited_in_current_path:  # Prevent cycles
                find_all_paths_dfs(src_task_graph, neighbor_id, end_node_id, current_path, all_paths, visited_in_current_path)
    # Backtrack: remove current node from path and visited set
    current_path.pop()
    visited_in_current_path.remove(current_node_id)


def find_all_paths(src_task_graph: TaskGraph, start_node_id: str, end_node_id: str) -> List[List[str]]:
    """
    Finds all simple paths from a start node to an end node in a TaskGraph.

    Args:
        src_task_graph (TaskGraph): The TaskGraph to search within.
        start_node_id (str): The ID of the starting node.
        end_node_id (str): The ID of the ending node.

    Returns:
        List[List[str]]: A list of lists, where each inner list represents a path
                         as a sequence of node IDs.
    """
    if start_node_id not in src_task_graph.nodes or end_node_id not in src_task_graph.nodes:
        logger.warning(f"Pathfinding: Start '{start_node_id}' or end '{end_node_id}' not in graph '{src_task_graph.graph_id}'.")
        return []

    all_paths: List[List[str]] = []
    src_task_graph.build_adjacency_lists()  # Ensure adj lists are ready for DFS
    find_all_paths_dfs(src_task_graph, start_node_id, end_node_id, [], all_paths, set())
    return all_paths


def split_single_structure(src_task_graph: TaskGraph, parallel_branch_node_lists: List[List[str]], merge_node_id: str, base_subgraph_idx_str: str) -> List[TaskGraph]:
    """
    Splits a graph based on identified parallel branches converging at a merge node.
    Each branch, along with its common upstream nodes and the common downstream nodes
    starting from the merge node, forms a new subgraph.

    Args:
        src_task_graph (TaskGraph): The original TaskGraph to split.
        parallel_branch_node_lists (List[List[str]]): A list of lists, where each inner list
                                                      represents the sequence of nodes for a specific branch
                                                      leading up to the merge node (exclusive of merge node).
        merge_node_id (str): The ID of the node where the parallel branches re-converge.
        base_subgraph_idx_str (str): A string prefix for naming the generated subgraphs.

    Returns:
        List[TaskGraph]: A list of new TaskGraph objects, each representing one of the split branches.
    """
    if not src_task_graph.nodes:
        return []
    if merge_node_id not in src_task_graph.nodes:
        return []

    src_task_graph.build_adjacency_lists()

    # Identify all nodes common to all branches from the merge node downwards
    common_downstream_nodes_ids: Set[str] = get_all_downstream_nodes_recursive(src_task_graph, merge_node_id, visited=set())
    if not common_downstream_nodes_ids:
        common_downstream_nodes_ids = {merge_node_id}  # If merge node has no children, it's still part of the common suffix

    created_subgraphs: List[TaskGraph] = []
    # Create a subgraph for each identified parallel branch
    for i, branch_nodes_prefix in enumerate(parallel_branch_node_lists):
        subgraph_id: str = f"{src_task_graph.graph_id}_{base_subgraph_idx_str}_b{i + 1}"
        subgraph: TaskGraph = TaskGraph(graph_id=subgraph_id)
        current_subgraph_node_ids: Set[str] = set()

        # Add nodes from the unique part of the branch
        for node_id in branch_nodes_prefix:
            if node_id == merge_node_id:
                continue  # Merge node is added with common_downstream_nodes_ids
            if node_id not in src_task_graph.nodes:
                current_subgraph_node_ids.clear()
                break  # Invalid node in branch, skip this subgraph
            subgraph.add_node(copy.deepcopy(src_task_graph.nodes[node_id]))
            current_subgraph_node_ids.add(node_id)

        if not current_subgraph_node_ids and branch_nodes_prefix:
            continue  # If branch_nodes_prefix was not empty but no valid nodes were added

        # Add common downstream nodes (including the merge node) to the subgraph
        for node_id in common_downstream_nodes_ids:
            if node_id not in src_task_graph.nodes:
                current_subgraph_node_ids.clear()
                break  # Invalid node, skip this subgraph
            if node_id not in current_subgraph_node_ids:
                subgraph.add_node(copy.deepcopy(src_task_graph.nodes[node_id]))
                current_subgraph_node_ids.add(node_id)

        if not current_subgraph_node_ids:
            continue  # Skip if no nodes were added to the subgraph

        # Adjust dependencies for nodes within the new subgraph to only refer to other nodes in the same subgraph
        valid_subgraph_nodes: Set[str] = set(subgraph.nodes.keys())
        for sg_node_id in list(subgraph.nodes.keys()):
            original_node: Optional[Node] = src_task_graph.nodes.get(sg_node_id)
            if not original_node:
                continue
            new_deps: List[str] = [dep for dep in original_node.dependencies if dep in valid_subgraph_nodes]
            subgraph.nodes[sg_node_id].dependencies = new_deps

        subgraph.build_adjacency_lists()  # Rebuild adj lists for the new subgraph
        is_valid, msg = subgraph.validate_graph()

        if is_valid and subgraph.nodes:
            created_subgraphs.append(subgraph)
        elif subgraph.nodes:
            logger.error(f"Invalid reconverge subgraph '{subgraph.graph_id}': {msg}.")
        else:
            logger.warning(f"Empty reconverge subgraph '{subgraph.graph_id}'.")

    return created_subgraphs


def split_by_fan_out_to_exits(src_task_graph: TaskGraph, naming_prefix_idx: int) -> List[TaskGraph]:
    """
    Attempts to split a TaskGraph if it contains a fan-out node leading to multiple
    distinct exit nodes that do not re-converge.

    Args:
        src_task_graph (TaskGraph): The TaskGraph to analyze and potentially split.
        naming_prefix_idx (int): An index used for unique naming of generated subgraphs.

    Returns:
        List[TaskGraph]: A list of new TaskGraph objects if a split occurs, otherwise an empty list.
    """
    src_task_graph.build_adjacency_lists()
    if not src_task_graph.nodes:
        return []

    is_valid, msg = src_task_graph.validate_graph()  # Validate before proceeding
    if not is_valid:
        logger.error(f"Fan-out: Invalid graph '{src_task_graph.graph_id}': {msg}.")
        return []

    original_exit_node_ids: Set[str] = {n.node_id for n in src_task_graph.get_exit_nodes()}
    if len(original_exit_node_ids) <= 1:
        return []  # No fan-out to multiple distinct exits possible if only one or no exits

    # Iterate through each node to find potential fork points
    for fork_candidate_id in sorted(list(src_task_graph.nodes.keys())):
        children_ids: List[str] = sorted(list(src_task_graph.adj.get(fork_candidate_id, [])))
        if len(children_ids) < 2:
            continue  # A fork point must have at least two children

        child_branch_details: List[Tuple[str, Set[str], Set[str]]] = []
        # For each child of the fork, determine its downstream nodes and reachable exit nodes
        for child_id in children_ids:
            downstream_of_child: Set[str] = get_all_downstream_nodes_recursive(src_task_graph, child_id, visited=set())
            reachable_exits: Set[str] = downstream_of_child.intersection(original_exit_node_ids)
            if reachable_exits:
                child_branch_details.append((child_id, downstream_of_child, reachable_exits))

        if len(child_branch_details) < 2:
            continue  # Need at least two branches with reachable exits

        # Check combinations of branches to find pairwise disjoint exit sets
        for r in range(len(child_branch_details), 1, -1):  # Start with largest combos, then smaller
            for combo in itertools.combinations(child_branch_details, r):
                exit_sets_in_combo: List[Set[str]] = [details[2] for details in combo]
                is_pairwise_disjoint: bool = True
                temp_union_of_exits: Set[str] = set()
                # Verify that the exit nodes for the chosen branches are mutually exclusive
                for exits_set in exit_sets_in_combo:
                    if not exits_set.isdisjoint(temp_union_of_exits):
                        is_pairwise_disjoint = False
                        break
                    temp_union_of_exits.update(exits_set)

                if is_pairwise_disjoint and len(temp_union_of_exits) >= r:  # Ensure there are at least 'r' distinct exits
                    # Collect all ancestors of the fork node and the fork node itself
                    ancestors_of_fork: Set[str] = get_all_ancestors(src_task_graph, fork_candidate_id)
                    common_upstream_nodes: Set[str] = ancestors_of_fork | {fork_candidate_id}

                    current_split_generated_graphs: List[TaskGraph] = []
                    # Create a new subgraph for each branch in the disjoint combo
                    for i, (child_c, downstream_nodes_c, exits_c) in enumerate(combo):
                        subgraph_node_ids: Set[str] = common_upstream_nodes | downstream_nodes_c
                        subgraph: TaskGraph = TaskGraph(graph_id=f"{src_task_graph.graph_id}_fan{naming_prefix_idx}_{fork_candidate_id}_b{i + 1}")

                        # Add relevant nodes to the new subgraph
                        for node_id_to_add in subgraph_node_ids:
                            if node_id_to_add in src_task_graph.nodes:
                                subgraph.add_node(copy.deepcopy(src_task_graph.nodes[node_id_to_add]))

                        if not subgraph.nodes:
                            continue

                        # Adjust dependencies within the new subgraph
                        for sg_node_id_adj in list(subgraph.nodes.keys()):
                            original_node_deps: List[str] = src_task_graph.nodes[sg_node_id_adj].dependencies
                            new_deps_adj: List[str] = [dep for dep in original_node_deps if dep in subgraph.nodes]
                            subgraph.nodes[sg_node_id_adj].dependencies = new_deps_adj

                        subgraph.build_adjacency_lists()  # Rebuild adj lists for the new subgraph
                        is_sg_valid, sg_msg = subgraph.validate_graph()

                        if is_sg_valid and subgraph.nodes:
                            current_split_generated_graphs.append(subgraph)
                        elif subgraph.nodes:
                            logger.error(f"Invalid fanout subgraph '{subgraph.graph_id}': {sg_msg}.")

                    if current_split_generated_graphs:  # If any valid graphs were made for this combo
                        logger.info(f"Fan-out split at '{fork_candidate_id}' (graph '{src_task_graph.graph_id}') -> {len(current_split_generated_graphs)} subgraphs.")
                        return current_split_generated_graphs  # Return the first successful split found
    return []  # No fan-out split found for the entire graph


def split_by_reconverging_paths(src_task_graph: TaskGraph, naming_prefix_idx: int) -> List[TaskGraph]:
    """
    Attempts to split a TaskGraph if it contains re-converging parallel paths.
    It identifies common merge points and splits the graph into subgraphs,
    each representing a unique path leading to the merge point plus the common suffix.

    Args:
        src_task_graph (TaskGraph): The TaskGraph to analyze and potentially split.
        naming_prefix_idx (int): An index used for unique naming of generated subgraphs.

    Returns:
        List[TaskGraph]: A list of new TaskGraph objects if a split occurs, otherwise an empty list.
    """
    src_task_graph.build_adjacency_lists()
    if not src_task_graph.nodes:
        return []

    is_valid, msg = src_task_graph.validate_graph()
    if not is_valid:
        logger.error(f"Reconv: Invalid graph '{src_task_graph.graph_id}': {msg}.")
        return []

    entry_node_ids: List[str] = [n.node_id for n in src_task_graph.get_entry_nodes()]
    exit_node_ids: List[str] = [n.node_id for n in src_task_graph.get_exit_nodes()]
    if not entry_node_ids or not exit_node_ids:
        return []

    all_e2e_paths: List[List[str]] = []
    path_tuples_seen: Set[Tuple[str, ...]] = set()
    # Find all unique end-to-end paths in the graph
    for e_id in entry_node_ids:
        for x_id in exit_node_ids:
            paths: List[List[str]] = find_all_paths(src_task_graph, e_id, x_id)
            for p in paths:
                pt: Tuple[str, ...] = tuple(p)
                if pt not in path_tuples_seen:
                    all_e2e_paths.append(p)
                    path_tuples_seen.add(pt)

    if len(all_e2e_paths) <= 1:
        return []  # No parallel paths to consider for reconvergence

    found_split_candidates: List[Tuple[List[List[str]], str]] = []  # Stores (list of branch node sequences, merge node ID)
    # Compare all pairs of paths to find common suffixes and diverging prefixes
    for i in range(len(all_e2e_paths)):
        for j in range(i + 1, len(all_e2e_paths)):
            path1, path2 = all_e2e_paths[i], all_e2e_paths[j]
            merge_idx_p1, merge_idx_p2, first_common_node_in_suffix = -1, -1, None
            # Traverse paths backward to find the first common node (merge point)
            for k_idx_from_end in range(min(len(path1), len(path2))):
                node_p1: str = path1[len(path1) - 1 - k_idx_from_end]
                node_p2: str = path2[len(path2) - 1 - k_idx_from_end]
                if node_p1 == node_p2:
                    first_common_node_in_suffix = node_p1
                    merge_idx_p1, merge_idx_p2 = len(path1) - 1 - k_idx_from_end, len(path2) - 1 - k_idx_from_end
                else:
                    break  # Paths diverge

            # If a common merge node is found and branches are not empty/identical
            if first_common_node_in_suffix is not None and merge_idx_p1 > 0 and merge_idx_p2 > 0:
                branch1_nodes: List[str] = path1[:merge_idx_p1]
                branch2_nodes: List[str] = path2[:merge_idx_p2]
                if not branch1_nodes or not branch2_nodes or tuple(branch1_nodes) == tuple(branch2_nodes):
                    continue

                # Group branches that merge into the same node
                is_new_candidate_group: bool = True
                for cand_idx, (existing_branches_list, existing_merge_node) in enumerate(found_split_candidates):
                    if existing_merge_node == first_common_node_in_suffix:
                        is_new_candidate_group = False
                        current_branch_tuples: Set[Tuple[str, ...]] = {tuple(b) for b in existing_branches_list}
                        if tuple(branch1_nodes) not in current_branch_tuples:
                            existing_branches_list.append(branch1_nodes)
                        if tuple(branch2_nodes) not in current_branch_tuples:
                            existing_branches_list.append(branch2_nodes)
                        break
                if is_new_candidate_group:
                    found_split_candidates.append(([branch1_nodes, branch2_nodes], first_common_node_in_suffix))

    if not found_split_candidates:
        return []

    all_resulting_subgraphs: List[TaskGraph] = []
    processed_split_signatures_for_subgraph_gen: Set[Tuple[Tuple[Tuple[str, ...], ...], str]] = set()

    # Process each found split candidate to generate subgraphs
    for split_idx, (branch_definitions, merge_node) in enumerate(found_split_candidates):
        if len(branch_definitions) < 2:
            continue  # Need at least two branches for a split

        # Sort branches for canonical signature to avoid redundant processing
        sorted_branch_tuples: Tuple[Tuple[str, ...], ...] = tuple(sorted([tuple(b) for b in branch_definitions]))
        current_split_signature: Tuple[Tuple[Tuple[str, ...], ...], str] = (sorted_branch_tuples, merge_node)

        if current_split_signature in processed_split_signatures_for_subgraph_gen:
            continue
        processed_split_signatures_for_subgraph_gen.add(current_split_signature)

        naming_str: str = f"reconv{naming_prefix_idx}_{merge_node}_s{split_idx}"
        subgraphs_from_this_split: List[TaskGraph] = split_single_structure(src_task_graph, branch_definitions, merge_node, naming_str)
        if subgraphs_from_this_split:  # Only extend if non-empty
            logger.info(f"Re-converge split for merge '{merge_node}' (graph '{src_task_graph.graph_id}') -> {len(subgraphs_from_this_split)} subgraphs.")
            all_resulting_subgraphs.extend(subgraphs_from_this_split)

    return all_resulting_subgraphs


def discover_and_split_parallel_paths(src_task_graph: TaskGraph) -> List[TaskGraph]:
    """
    Discovers and splits a TaskGraph into irreducible subgraphs by iteratively identifying
    and splitting fan-out and re-converging parallel paths.

    Args:
        src_task_graph (TaskGraph): The original TaskGraph to be analyzed and split.

    Returns:
        List[TaskGraph]: A list of TaskGraph objects, where each represents an
                         irreducible (cannot be further split by these rules) subgraph.
    """
    if not src_task_graph or not src_task_graph.nodes:
        logger.info("Input graph is empty. Nothing to split.")
        return []

    initial_graph_copy: TaskGraph = src_task_graph.copy() if hasattr(src_task_graph, "copy") else copy.deepcopy(src_task_graph)
    initial_graph_copy.build_adjacency_lists()  # Ensure it's ready for validation
    is_valid, msg = initial_graph_copy.validate_graph()
    if not is_valid:
        logger.error(f"Original graph '{initial_graph_copy.graph_id}' is invalid: {msg}. Cannot split.")
        return [initial_graph_copy]

    final_irreducible_graphs: List[TaskGraph] = []
    processing_queue: List[TaskGraph] = [initial_graph_copy]
    # Use structural signature to track processed graphs and avoid redundant work on identical structures
    processed_structural_signatures_in_queue: Set[str] = set()

    iteration_counter: int = 0  # For unique naming of intermediate graphs

    # Process graphs in a queue until no more splits can be made
    while processing_queue:
        current_graph: TaskGraph = processing_queue.pop(0)
        current_graph_structural_sig: str = generate_structural_signature(current_graph)

        if current_graph_structural_sig in processed_structural_signatures_in_queue:
            logger.debug(f"Skipping already processed graph structure (sig: {current_graph_structural_sig[:70]}...) for graph ID {current_graph.graph_id}")
            continue
        # Add to processed signatures AFTER successful splitting attempt (or if it's irreducible)
        # processed_structural_signatures_in_queue.add(current_graph_structural_sig) # Moved this to after potential splits

        iteration_counter += 1
        split_occurred_this_pass: bool = False

        # 1. Try to split by fan-out to distinct exits
        graphs_after_fan_out: List[TaskGraph] = split_by_fan_out_to_exits(current_graph, iteration_counter)
        if graphs_after_fan_out:  # Non-empty list means split occurred
            for g_fan in graphs_after_fan_out:
                # Add newly created subgraphs to the queue for further processing
                processing_queue.append(g_fan)
            split_occurred_this_pass = True

        if split_occurred_this_pass:
            logger.debug(f"Graph '{current_graph.graph_id}' processed with fan-out. Re-evaluating queue.")
            # If a split occurred, the current graph is replaced by its subgraphs.
            # Mark its structural signature as processed so it's not re-added
            processed_structural_signatures_in_queue.add(current_graph_structural_sig)
            continue  # Continue to next graph in queue

        # 2. If no fan-out split, try to split by re-converging paths
        graphs_after_reconverge: List[TaskGraph] = split_by_reconverging_paths(current_graph, iteration_counter)
        if graphs_after_reconverge:  # Non-empty list means split occurred
            for g_reconv in graphs_after_reconverge:
                # Add newly created subgraphs to the queue for further processing
                processing_queue.append(g_reconv)
            split_occurred_this_pass = True

        if split_occurred_this_pass:
            logger.debug(f"Graph '{current_graph.graph_id}' processed with re-convergence. Re-evaluating queue.")
            # If a split occurred, the current graph is replaced by its subgraphs.
            # Mark its structural signature as processed so it's not re-added
            processed_structural_signatures_in_queue.add(current_graph_structural_sig)
            continue  # Continue to next graph in queue

        # If no split of any type happened on current_graph, it's irreducible
        if not split_occurred_this_pass:
            logger.info(f"Graph '{current_graph.graph_id}' (struct_sig: {current_graph_structural_sig[:70]}...) is irreducible. Adding to final list.")
            final_irreducible_graphs.append(current_graph)
            # Mark its structural signature as processed only when it is declared irreducible
            processed_structural_signatures_in_queue.add(current_graph_structural_sig)

    # Deduplicate final list of graphs based on structure and assign canonical names
    unique_final_graphs_map: Dict[str, TaskGraph] = {}
    true_final_graphs: List[TaskGraph] = []
    base_id_for_final_naming: str = src_task_graph.graph_id

    for g in final_irreducible_graphs:
        structural_sig: str = generate_structural_signature(g)
        if structural_sig not in unique_final_graphs_map:
            new_final_id: str = f"{base_id_for_final_naming}_split_{len(unique_final_graphs_map) + 1}"
            g.graph_id = new_final_id  # Update graph_id for the returned unique graph
            unique_final_graphs_map[structural_sig] = g
            true_final_graphs.append(g)
        else:
            logger.debug(f"Skipping structurally duplicate final graph: current id {g.graph_id} (already found as {unique_final_graphs_map[structural_sig].graph_id}).")

    logger.info(f"Original graph '{src_task_graph.graph_id}' resulted in {len(true_final_graphs)} unique irreducible TaskGraph(s).")
    return true_final_graphs


if __name__ == "__main__":
    # Ensure a directory for DAG images exists
    if not os.path.exists("dag_images"):
        os.makedirs("dag_images")

    # Example 1: Re-converging paths
    logger.info(f"\n--- Splitting graph: ex1_reconverge ---")
    node_a = Node(node_id="A", node_type=NodeType.DATA_LOAD)
    node_b = Node(node_id="B", node_type=NodeType.COMPUTE, dependencies=["A"])
    node_a1 = Node(node_id="A1", node_type=NodeType.DATA_LOAD)
    node_b1 = Node(node_id="B1", node_type=NodeType.COMPUTE, dependencies=["A1"])
    node_c = Node(node_id="C", node_type=NodeType.COMPUTE, dependencies=["B", "B1"])  # Re-convergence point
    node_d_ex1 = Node(node_id="D_ex1", node_type=NodeType.COMPUTE, dependencies=["C"])
    node_e_ex1 = Node(node_id="E_ex1", node_type=NodeType.MODEL_TRAIN, dependencies=["D_ex1"])

    original_graph_ex1 = TaskGraph(graph_id="ex1_reconverge")
    original_graph_ex1.add_nodes([node_a, node_b, node_a1, node_b1, node_c, node_d_ex1, node_e_ex1])
    if original_graph_ex1.nodes:
        original_graph_ex1.save_dag_pic(filename=original_graph_ex1.graph_id + "_orig_pic", directory="dag_images")

    split_graphs1 = discover_and_split_parallel_paths(original_graph_ex1)
    for idx, sg in enumerate(split_graphs1):
        logger.info(f"Final Subgraph {idx + 1}: {sg.graph_id} with {len(sg.nodes)} nodes.")
        if sg.nodes:
            sg.save_dag_pic(filename=sg.graph_id + "_pic", directory="dag_images")
    logger.info(f"--- Finished Ex1 ---\n")

    # Example 2: Complex (fan-out and re-converging)
    logger.info(f"\n--- Splitting graph: ex2_complex ---")
    node_x = Node("X", NodeType.DATA_LOAD)
    node_y = Node("Y", NodeType.DATA_LOAD)
    node_p1 = Node("P1", NodeType.COMPUTE, dependencies=["X"])
    node_p2 = Node("P2", NodeType.COMPUTE, dependencies=["Y"])
    node_m1 = Node("M1", NodeType.COMPUTE, dependencies=["P1", "P2"])  # Re-convergence 1
    node_p3 = Node("P3", NodeType.DATA_LOAD)
    node_z = Node("Z", NodeType.COMPUTE, dependencies=["M1", "P3"])  # Re-convergence 2 (Z depends on P3 and M1 which itself is a merge)
    node_j1 = Node("J1", NodeType.COMPUTE, dependencies=["Z"])
    node_j2 = Node("J2", NodeType.COMPUTE, dependencies=["Z"])
    node_k1 = Node("K1", NodeType.MODEL_TRAIN, dependencies=["J1"])  # Exit 1
    node_k2 = Node("K2", NodeType.MODEL_TRAIN, dependencies=["J2"])  # Exit 2 (Fan-out at Z to J1, J2 leading to distinct exits K1, K2)

    complex_graph_ex2 = TaskGraph(graph_id="ex2_complex")
    complex_graph_ex2.add_nodes([node_x, node_y, node_p1, node_p2, node_m1, node_p3, node_z, node_j1, node_j2, node_k1, node_k2])
    if complex_graph_ex2.nodes:
        complex_graph_ex2.save_dag_pic(filename=complex_graph_ex2.graph_id + "_orig_pic", directory="dag_images")

    split_graphs2 = discover_and_split_parallel_paths(complex_graph_ex2)
    for idx, sg in enumerate(split_graphs2):
        logger.info(f"Final Subgraph {idx + 1}: {sg.graph_id} with {len(sg.nodes)} nodes.")
        if sg.nodes:
            sg.save_dag_pic(filename=sg.graph_id + "_pic", directory="dag_images")
    logger.info(f"--- Finished Ex2 ---\n")

    # Example 3: Simple linear graph (should not split)
    logger.info(f"\n--- Splitting graph: ex3_linear ---")
    linear_graph_ex3 = TaskGraph(graph_id="ex3_linear")
    linear_graph_ex3.add_nodes([Node("L1", NodeType.DATA_LOAD), Node("L2", NodeType.COMPUTE, dependencies=["L1"]), Node("L3", NodeType.MODEL_TRAIN, dependencies=["L2"])])
    if linear_graph_ex3.nodes:
        linear_graph_ex3.save_dag_pic(filename=linear_graph_ex3.graph_id + "_orig_pic", directory="dag_images")
    split_graphs3 = discover_and_split_parallel_paths(linear_graph_ex3)
    for idx, sg in enumerate(split_graphs3):
        logger.info(f"Final Subgraph {idx + 1}: {sg.graph_id} ({len(sg.nodes)} nodes)")
        if sg.nodes:
            sg.save_dag_pic(filename=sg.graph_id + "_pic", directory="dag_images")
    logger.info(f"--- Finished Ex3 ---\n")

    # Example 4: Fan-out, no re-merge (should split into two distinct paths)
    logger.info(f"\n--- Splitting graph: ex4_fanout_only ---")
    split_no_merge_graph_ex4 = TaskGraph(graph_id="ex4_fanout_only")
    split_no_merge_graph_ex4.add_nodes(
        [
            Node("S_A", NodeType.DATA_LOAD),
            Node("S_B_exit1", NodeType.COMPUTE, dependencies=["S_A"]),  # Path 1
            Node("S_C_exit2", NodeType.COMPUTE, dependencies=["S_A"]),  # Path 2
        ]
    )
    if split_no_merge_graph_ex4.nodes:
        split_no_merge_graph_ex4.save_dag_pic(filename=split_no_merge_graph_ex4.graph_id + "_orig_pic", directory="dag_images")
    split_graphs4 = discover_and_split_parallel_paths(split_no_merge_graph_ex4)
    for idx, sg in enumerate(split_graphs4):
        logger.info(f"Final Subgraph {idx + 1}: {sg.graph_id} ({len(sg.nodes)} nodes)")
        if sg.nodes:
            sg.save_dag_pic(filename=sg.graph_id + "_pic", directory="dag_images")
    logger.info(f"--- Finished Ex4 ---\n")

    # Example 5: Three-way re-merge (should split into three paths)
    logger.info(f"\n--- Splitting graph: ex5_3way_reconverge ---")
    three_way_graph_ex5 = TaskGraph(graph_id="ex5_3way_reconverge")
    three_way_graph_ex5.add_nodes(
        [
            Node("3W_A", NodeType.DATA_LOAD),
            Node("3W_B", NodeType.DATA_LOAD),
            Node("3W_E", NodeType.DATA_LOAD),
            Node("3W_C", NodeType.COMPUTE, dependencies=["3W_A", "3W_B", "3W_E"]),  # 3-way re-convergence
            Node("3W_D_exit", NodeType.MODEL_TRAIN, dependencies=["3W_C"]),
        ]
    )
    if three_way_graph_ex5.nodes:
        three_way_graph_ex5.save_dag_pic(filename=three_way_graph_ex5.graph_id + "_orig_pic", directory="dag_images")
    split_graphs5 = discover_and_split_parallel_paths(three_way_graph_ex5)
    for idx, sg in enumerate(split_graphs5):
        logger.info(f"Final Subgraph {idx + 1}: {sg.graph_id} ({len(sg.nodes)} nodes)")
        if sg.nodes:
            sg.save_dag_pic(filename=sg.graph_id + "_pic", directory="dag_images")
    logger.info(f"--- Finished Ex5 ---\n")

    logger.info("All examples processed. Check the 'dag_images' folder.")
