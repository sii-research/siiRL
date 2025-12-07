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
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from siirl.execution.dag.node import Node, NodeRole, NodeStatus, NodeType


class TaskGraph:
    """
    Represents a Directed Acyclic Graph (DAG) of tasks, composed of multiple Node objects and their dependencies.
    """

    def __init__(self, graph_id: str):
        """
        Initialize a task graph.
        Parameters:
            graph_id (str): The unique identifier of the graph.
        """
        self.graph_id: str = graph_id
        self.nodes: Dict[str, Node] = {}  # node_id -> Node object
        # Forward adjacency list: node_id -> list of node_ids that depend on it (dependents)
        self.adj: Dict[str, List[str]] = {}
        # Reverse adjacency list (more commonly used for dependency checking):
        #       node_id -> list of node_ids it depends on (dependencies)
        self.rev_adj: Dict[str, List[str]] = {}

    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph.
        If the node already exists, it will be updated (Note: This may cause inconsistent dependencies.
            It is recommended to delete and then add, or use with caution).
        Parameters:
            node (Node): The Node object to add.
        """
        if not isinstance(node, Node):
            raise ValueError("Only Node type objects can be added to the graph.")
        if node.node_id in self.nodes:
            logger.warning(f"Warning: Node {node.node_id} already exists in the graph. It will be replaced.")

        self.nodes[node.node_id] = node
        # Update the adjacency list
        self._update_adj_for_node(node)

    def add_nodes(self, nodes: List[Node]) -> None:
        """
        Add multiple nodes to the graph in batch.
        If a node already exists, it will be updated (Note: This may cause inconsistent dependencies.
            It is recommended to delete and then add, or use with caution).
        Parameters:
            nodes (List[Node]): A list of Node objects to add.
        """
        for node in nodes:
            self.add_node(node)

    def _update_adj_for_node(self, node: Node) -> None:
        """Update the adjacency list and reverse adjacency list for a single node."""
        # Initialize the adjacency list for the current node
        self.adj.setdefault(node.node_id, [])
        self.rev_adj.setdefault(node.node_id, [])

        # Update the reverse adjacency list (node -> its dependencies)
        self.rev_adj[node.node_id] = list(node.dependencies)  # Ensure it's a copy of the list

        # Update the forward adjacency list (dependency -> node)
        for dep_id in node.dependencies:
            if dep_id not in self.nodes:
                # Allow adding nodes whose dependencies do not yet exist,
                # but it will be checked when validating the graph
                pass
            self.adj.setdefault(dep_id, []).append(node.node_id)
            # Remove duplicates, although there should usually be no duplicate dependencies
            self.adj[dep_id] = list(set(self.adj[dep_id]))

    def build_adjacency_lists(self) -> None:
        """
        Completely (re)build the adjacency list and reverse adjacency list based on
            the dependencies of all nodes in the graph.
        Call this method after all nodes have been added,
            or after significant changes to the node dependencies.
        """
        self.adj.clear()
        self.rev_adj.clear()

        for node_id, node in self.nodes.items():
            self.adj.setdefault(node_id, [])
            self.rev_adj.setdefault(node_id, list(node.dependencies))  # node -> its dependencies
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    # Allow dependencies on nodes not yet defined in the graph. validate_graph will handle this.
                    self.adj.setdefault(dep_id, [])  # Ensure dep_id is in adj
                else:
                    self.adj.setdefault(dep_id, []).append(node_id)  # dependency -> node

        # Clean up duplicates in adj (if necessary)
        for node_id in self.adj:
            self.adj[node_id] = list(set(self.adj[node_id]))

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get the node with the specified ID.
        Parameters:
            node_id (str): The node ID.
        Returns:
            Optional[Node]: The Node object if found, otherwise None.
        """
        return self.nodes.get(node_id)

    def get_dependencies(self, node_id: str) -> List[Node]:
        """
        Get all direct dependent nodes of the specified node.
        Parameters:
            node_id (str): The node ID.
        Returns:
            List[Node]: A list of dependent Node objects.
        """
        if node_id not in self.nodes:
            return []
        # return [self.nodes[dep_id] for dep_id in self.nodes[node_id].dependencies if dep_id in self.nodes]
        # It's more straightforward to use rev_adj if it's already built
        if not self.rev_adj or node_id not in self.rev_adj:  # Ensure rev_adj is built
            self.build_adjacency_lists()

        return [self.nodes[dep_id] for dep_id in self.rev_adj.get(node_id, []) if dep_id in self.nodes]

    def get_dependents(self, node_id: str) -> List[Node]:
        """
        Get all nodes that directly depend on the specified node.
        Parameters:
            node_id (str): The node ID.
        Returns:
            List[Node]: A list of Node objects that depend on this node.
        """
        if node_id not in self.nodes:
            return []
        if not self.adj or node_id not in self.adj:  # Ensure adj is built
            self.build_adjacency_lists()

        return [self.nodes[dep_id] for dep_id in self.adj.get(node_id, []) if dep_id in self.nodes]

    def get_downstream_nodes(self, node_id: str) -> List[Node]:
        """Get the direct downstream nodes of a node"""
        return self.get_dependents(node_id)

    def get_entry_nodes(self) -> List[Node]:
        """
        Get all entry nodes in the graph that have no dependencies.
        Returns:
            List[Node]: A list of entry Node objects.
        """
        if not self.rev_adj:  # Ensure rev_adj is built
            self.build_adjacency_lists()
        return [node for node_id, node in self.nodes.items() if not self.rev_adj.get(node_id)]

    def get_exit_nodes(self) -> List[Node]:
        """
        Get all exit nodes in the graph that have no subsequent dependent nodes.
        Returns:
            List[Node]: A list of exit Node objects.
        """
        if not self.adj:  # Ensure adj is built
            self.build_adjacency_lists()
        return [node for node_id, node in self.nodes.items() if not self.adj.get(node_id)]

    def validate_graph(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the validity of the graph.
        1. Check if all node dependencies exist in the graph.
        2. Check if there are circular dependencies (determine if it's a DAG).
        Returns:
            Tuple[bool, Optional[str]]: (Is valid, Error message or None)
        """
        # 1. Check dependency existence
        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    return False, f"The dependency '{dep_id}' of node '{node_id}' does not exist in the graph."

        # 2. Check for circular dependencies (using topological sorting)
        try:
            self.get_topological_sort()
            return True, None
        except ValueError as e:  # get_topological_sort will raise a ValueError when a cycle is detected
            return False, str(e)

    def get_topological_sort(self) -> List[str]:
        """
        Get the topological sorting of the nodes.
        If the graph is not a DAG (i.e., there is a cycle), a ValueError will be raised.
        Use Kahn's algorithm.
        Returns:
            List[str]: A list of node IDs in topological order.
        """
        if not self.nodes:
            return []

        # Ensure the adjacency list is up-to-date
        self.build_adjacency_lists()

        in_degree = {node_id: 0 for node_id in self.nodes}
        for node_id in self.nodes:
            for dependent_id in self.adj.get(node_id, []):
                in_degree[dependent_id] += 1

        # In some cases, if a node is in the keys of adj but not in nodes
        # (e.g., a dependency is declared but not added as a node itself),
        # in_degree may contain keys that are not in self.nodes. Ensure only nodes in the graph are processed.
        # However, it's better to handle this in build_adjacency_lists or add_node.
        # Here, it's assumed that the keys of in_degree are all valid nodes in self.nodes.

        queue = [node_id for node_id in self.nodes if in_degree[node_id] == 0]
        topological_order = []

        while queue:
            u = queue.pop(0)
            topological_order.append(u)

            for v_id in self.adj.get(u, []):  # v_id is a subsequent node of u
                if v_id in in_degree:  # Ensure v_id is a node in the graph
                    in_degree[v_id] -= 1
                    if in_degree[v_id] == 0:
                        queue.append(v_id)
                # else: # If v_id is not in in_degree, it may be a dependency not defined in nodes
                #     pass # validate_graph should have already captured this situation

        if len(topological_order) != len(self.nodes):
            # Find the nodes in the cycle (optional, more complex)
            # For simplicity, only report the existence of a cycle
            raise ValueError(f"There are circular dependencies in graph '{self.graph_id}', and topological sorting cannot be performed.")

        return topological_order

    def reset_nodes_status(self) -> None:
        """Reset the status of all nodes in the graph to PENDING, and clear the output and error information."""
        for node in self.nodes.values():
            node.status = NodeStatus.PENDING
            node.output = None
            node.error_info = None
            node.retries_done = 0

    @classmethod
    def load_from_config(cls, graph_id: str, config_data: List[Dict[str, Any]]) -> "TaskGraph":
        """
        Create a TaskGraph from configuration data (e.g., parsed from a YAML/JSON file).
        The configuration data should be a list of dictionaries, each describing a node.

        Parameters:
            graph_id (str): The ID of the graph.
            config_data (List[Dict[str, Any]]): A list of node configurations.
                Each dictionary should contain: 'node_id', 'node_type' (in string form),
                Optional: 'dependencies', 'config', 'executable_ref', 'node_role', 'retry_limit'.

        Returns:
            TaskGraph: The constructed task graph object.
        """
        graph = cls(graph_id)
        for node_conf in config_data:
            try:
                node_type_str = node_conf.get("node_type")
                if not node_type_str:
                    raise ValueError(f"Node configuration {node_conf.get('node_id', 'Unknown ID')} is missing 'node_type'.")

                node_type = NodeType[node_type_str.upper()]

                node_role_str = node_conf.get("node_role")
                node_role = NodeRole[node_role_str.upper()] if node_role_str else NodeRole.DEFAULT
                only_forward_compute = node_conf.get("only_forward_compute", False)
                agent_group = node_conf.get("agent_group", 0)

                node = Node(
                    node_id=node_conf["node_id"],
                    node_type=node_type,
                    node_role=node_role,
                    only_forward_compute=only_forward_compute,
                    agent_group=agent_group,
                    dependencies=node_conf.get("dependencies"),
                    config=node_conf.get("config"),
                    executable_ref=node_conf.get("executable_ref"),
                    retry_limit=node_conf.get("retry_limit", 0),
                )
                graph.add_node(node)
            except KeyError as e:
                raise ValueError(f"Node configuration {node_conf.get('node_id', 'Unknown ID')} is missing required field: {e}")
            except ValueError as e:  # e.g., NodeType['INVALID']
                raise ValueError(f"Node configuration {node_conf.get('node_id', 'Unknown ID')} has a value error: {e}")

        graph.build_adjacency_lists()  # Ensure the adjacency list is built after all nodes are added
        valid, msg = graph.validate_graph()
        if not valid:
            raise ValueError(f"The graph loaded from the configuration is invalid: {msg}")
        return graph

    def __repr__(self) -> str:
        """
        Return the topological structure of the DAG in text symbol form.
        Returns:
            str: A string representation of the DAG's topological structure.
        """
        output_lines = [f"TaskGraph(graph_id='{self.graph_id}', num_nodes={len(self.nodes)})"]
        if not self.nodes:
            return "\n".join(output_lines)

        try:
            # get_topological_sort will call build_adjacency_lists internally
            processing_order = self.get_topological_sort()
        except ValueError as e:  # Capture circular dependency errors
            return f"Unable to display DAG graph '{self.graph_id}': {e}"

        output_lines.append("=" * (len(output_lines[0])))

        for node_id in processing_order:
            node = self.nodes[node_id]
            output_lines.append(f"[{node.node_id}] ({node.node_type.value})")

            if node.executable_ref:
                output_lines.append(f"  Executable Ref: {node.executable_ref}")
            if node.config:
                output_lines.append(f"  Config: {node.config}")
            output_lines.append(f"        {node}")

            # Display upstream dependencies (parent nodes)
            parent_ids = sorted(self.rev_adj.get(node_id, []))
            if parent_ids:
                output_lines.append(f"  ↑ (Depends on upstream)")
                for parent_id in parent_ids:
                    output_lines.append(f"    ↖── [{parent_id}]")
            elif not parent_ids:  # It's an entry node
                output_lines.append("  (Entry node)")

            # Display downstream execution (child nodes)
            children_ids = sorted(self.adj.get(node_id, []))
            if children_ids:
                # output_lines.append(f"  ↓ (Subsequent execution)") # Optional title line
                for i, child_id in enumerate(children_ids):
                    child_node = self.nodes.get(child_id)  # The child node should exist
                    connector = "  └─→ " if i == len(children_ids) - 1 else "  ├─→ "
                    output_lines.append(f"{connector}[{child_id}] ({child_node.node_type.value if child_node else 'Unknown type'})")
            elif not children_ids:  # It's an exit node
                output_lines.append("  (Exit node)")

            output_lines.append("")  # Add a blank line after each node block for readability

        return "\n".join(output_lines).strip()

    def copy(self) -> "TaskGraph":
        new_graph = TaskGraph(graph_id=f"{self.graph_id}_copy")
        for _, original_node in self.nodes.items():
            new_graph.add_node(original_node.copy())
        new_graph.build_adjacency_lists()
        return new_graph

    def save_dag_pic(self, filename: str = "task_graph", directory: Optional[str] = None, view: bool = False, cleanup: bool = True) -> Optional[str]:
        """
        Visualize the DAG as an image using graphviz and save it.

        Parameters:
            filename (str): The file name of the output image (without the extension, e.g., "dag_pic").
                            The final file name will be filename.format (e.g., dag_pic.png).
            directory (Optional[str]): The directory to save the image.
                If None, it will be saved in the current working directory.
            view (bool): Whether to automatically open the image after generation.
            cleanup (bool): Whether to delete the temporary DOT source file after rendering.

        Returns:
            Optional[str]: The full path of the image if successful, otherwise None.
        """
        from graphviz import Digraph

        if not self.nodes:
            logger.warning(f"DAG graph '{self.graph_id}' is empty. No image will be generated.")
            return None

        # Ensure the adjacency list is up-to-date
        self.build_adjacency_lists()

        # Check the validity of the graph, e.g., if there are cycles
        is_valid, error_msg = self.validate_graph()
        if not is_valid:
            logger.error(f"Graph '{self.graph_id}' is invalid. Unable to generate image: {error_msg}")
            return None

        dot = Digraph(name=self.graph_id, comment=f"DAG for {self.graph_id}")
        dot.attr(rankdir="TB")  # Layout from top to bottom (optional LR: from left to right)
        dot.attr(label=f"DAG: {self.graph_id}", fontsize="20")
        dot.attr(labelloc="t")  # Title position at the top

        for node_id, node in self.nodes.items():
            # Node label contains ID, type, and role
            label = f"{node.node_id}\n({node.node_type.value})"
            if node.node_role:
                label += f"\nRole: {node.node_role.value}"

            colors = {NodeType.MODEL_TRAIN: "blue", NodeType.MODEL_INFERENCE: "green", NodeType.DATA_LOAD: "orange", NodeType.BARRIER_SYNC: "red"}
            color = colors.get(node.node_type, "black")
            node_attrs = {"penwidth": "2.0", "color": color, "fontcolor": color}

            dot.node(node_id, label=label, **node_attrs)

        # Add edges
        for node_id, children_ids in self.adj.items():
            # Skip dependency nodes not defined in the graph (theoretically should not happen)
            if node_id not in self.nodes:
                continue
            source_node = self.nodes[node_id]
            for child_id in children_ids:
                if child_id not in self.nodes:  # Skip child nodes not defined in the graph
                    continue
                # target_node = self.nodes[child_id]
                # Default edge color
                edge_color = "black"
                # If necessary, change the edge color according to the source or target node type, e.g.:
                # if source_node.node_type == NodeType.BARRIER_SYNC or target_node.node_type == NodeType.BARRIER_SYNC:
                #     edge_color = "red" # If you want the edges connected to Barrier to be red too
                dot.edge(node_id, child_id, color=edge_color)

        try:
            # Construct the full file path
            output_path = os.path.join(directory or ".", filename)
            rendered_path = dot.render(filename=output_path, directory=None, view=view, cleanup=cleanup, format="svg")
            logger.info(f"DAG image saved to: {rendered_path}")
            return rendered_path
        except Exception as e:
            logger.error(f"An error occurred while generating the DAG image: {e}. Please ensure the Graphviz executable is in your system PATH.")
            return None

    def get_nodes_by_type(self, node_types: List[NodeType]) -> List[Node]:
        """
        Retrieves all nodes from the graph that match any of the specified node types.

        Args:
            node_types: A list of NodeType enums to filter by.

        Returns:
            A list of Node objects whose type is in the node_types list.
        """
        return [node for node in self.nodes.values() if node.node_type in node_types]

    def get_nodes_by_role(self, node_role: NodeRole) -> List[Node]:
        """
        Retrieves all nodes from the graph that match the specified node role.

        Args:
            node_role: A NodeRole enum to filter by.

        Returns:
            A list of Node objects whose role matches the specified node_role.
        """
        return [node for node in self.nodes.values() if node.node_role == node_role]


# Example usage:
if __name__ == "__main__":
    logger.info("--- Demonstration of the core class of the DAG module ---")

    node_a = Node(node_id="rollout_actor", node_type=NodeType.MODEL_INFERENCE, node_role=NodeRole.ROLLOUT)
    node_b = Node(node_id="B", node_type=NodeType.MODEL_INFERENCE, node_role=NodeRole.ROLLOUT, dependencies=["A"])
    node_c = Node(node_id="C", node_type=NodeType.MODEL_INFERENCE, node_role=NodeRole.REFERENCE, dependencies=["A"])
    node_d = Node(node_id="D", node_type=NodeType.BARRIER_SYNC, dependencies=["B", "C"])
    node_e = Node(node_id="E", node_type=NodeType.MODEL_TRAIN, node_role=NodeRole.ACTOR, dependencies=["D"])

    graph = TaskGraph(graph_id="example_rl_pipeline")
    graph.add_nodes([node_a, node_b, node_c, node_d, node_e])
    graph.build_adjacency_lists()  # Ensure the adjacency list is built

    is_valid, validation_msg = graph.validate_graph()
    if is_valid:
        logger.info(f"\nGraph '{graph.graph_id}' passed validation.")
    else:
        logger.info(f"\nGraph '{graph.graph_id}' failed validation: {validation_msg}")
        exit(1)

    logger.info(f"{graph}")
    graph.save_dag_pic()

    logger.info("\n--- print_dag of a graph with multiple independent branches ---")
    multi_branch_graph = TaskGraph("multi_branch_dag")
    mb_n1 = Node("MB1", NodeType.DATA_LOAD)
    mb_n2 = Node("MB2", NodeType.COMPUTE, dependencies=["MB1"])
    mb_n3 = Node("MB3", NodeType.DATA_LOAD)  # Another entry node
    mb_n4 = Node("MB4", NodeType.COMPUTE, dependencies=["MB3"])
    mb_n5 = Node("MB5", NodeType.MODEL_TRAIN, dependencies=["MB2", "MB4"])  # Merge node
    multi_branch_graph.add_node(mb_n1)
    multi_branch_graph.add_node(mb_n2)
    multi_branch_graph.add_node(mb_n3)
    multi_branch_graph.add_node(mb_n4)
    multi_branch_graph.add_node(mb_n5)
    logger.info(multi_branch_graph)
    multi_branch_graph.save_dag_pic("multi_branch")

    logger.info("\n--- print_dag of an empty graph ---")
    empty_graph = TaskGraph("empty_graph_for_print_dag")
    logger.info(empty_graph)

    logger.info("\n--- print_dag of a graph with circular dependencies ---")
    cyclic_graph = TaskGraph("cyclic_graph_for_print_dag")
    cg_n1 = Node("CG1", NodeType.COMPUTE, dependencies=["CG2"])
    cg_n2 = Node("CG2", NodeType.COMPUTE, dependencies=["CG1"])
    cyclic_graph.add_node(cg_n1)
    cyclic_graph.add_node(cg_n2)
    logger.info(cyclic_graph)
