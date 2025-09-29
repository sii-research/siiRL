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

import json
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from siirl.execution.dag.node import Node, NodeRole, NodeType
from siirl.execution.dag.task_graph import TaskGraph


class Ref:
    """
    Represents the !Ref tag in YAML, used to reference other configuration paths.
    """

    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return f"!Ref {self.path}"


def ref_constructor(loader, node):
    """
    YAML constructor that parses the !Ref tag into a Ref object.
    Args:
        loader (yaml.Loader): The YAML loader instance.
        node (yaml.Node): The YAML node representing the !Ref tag.
    Returns:
        Ref: A Ref object containing the referenced path.
    """
    return Ref(loader.construct_scalar(node))


# Register custom YAML tag handlers
# but JSON does not support custom tags like !Ref directly.
yaml.SafeLoader.add_constructor("!Ref", ref_constructor)


def resolve_refs(config_item: Any, global_config: Dict[str, Any]) -> Any:
    """
    Parse the reference tag (!Ref) in the configuration item and replace it with the corresponding actual value
        in the global configuration.

    Args:
        config_item (Any): The configuration item to be parsed, which can be a
            dictionary, list, string, or other types.
        global_config (Dict[str, Any]): The global configuration dictionary
            containing all referenceable configuration items.

    Returns:
        Any: The parsed configuration item, where the !Ref tags have been replaced with actual values.

    Raises:
        ValueError: An exception is thrown when the referenced path does not exist in the global configuration.
    """

    if isinstance(config_item, dict):
        return {k: resolve_refs(v, global_config) for k, v in config_item.items()}
    elif isinstance(config_item, list):
        return [resolve_refs(item, global_config) for item in config_item]
    elif isinstance(config_item, Ref):
        ref_path = config_item.path
        parts = ref_path.split(".")
        if parts[0] != "global_config":
            raise ValueError(f"Unsupported yaml Ref 'parts[0]', only support 'global_config'")
        parts = parts[1:]
        current = global_config
        for part in parts:
            current = current.get(part)
            if current is None:
                raise ValueError(f"Unresolved reference '{ref_path}'.")
        return current
    else:
        return config_item


class DAGConfigLoader:
    """
    Loads, parses, and constructs TaskGraph objects from YAML or JSON files.
    """

    def __init__(self):
        pass

    @staticmethod
    def _parse_raw_config(raw_dag_config: Dict[str, Any], file_path: str) -> TaskGraph:
        """
        Helper function to parse and build a TaskGraph from a raw dictionary configuration.
        This function is called by load_dag_from_file to handle common logic after loading YAML/JSON.

        Args:
            raw_dag_config (Dict[str, Any]): The raw configuration dictionary loaded from the file.
            file_path (str): The path of the configuration file.

        Returns:
            TaskGraph: A TaskGraph object constructed from the configuration.
        """
        if not raw_dag_config:
            raise ValueError(f"The configuration file '{file_path}' is empty or has an incorrect format.")

        dag_id = raw_dag_config.get("dag_id")
        if not dag_id:
            raise ValueError(f"The 'dag_id' is missing in the configuration '{file_path}'.")

        description = raw_dag_config.get("description")
        global_config = raw_dag_config.get("global_config", {})

        # In YAML/JSON, nodes are defined as a list
        if "nodes" not in raw_dag_config:
            raise ValueError(f"The 'nodes' list is missing in the DAG configuration")
        nodes_list_config = raw_dag_config.get("nodes")
        if not isinstance(nodes_list_config, list):
            raise ValueError(f"The 'nodes' field in the configuration '{file_path}' must be a list.")

        dag_nodes: List[Node] = []
        node_ids = set()  # Used to store the node IDs that have appeared to verify uniqueness

        for i, node_config_dict in enumerate(nodes_list_config):
            if not isinstance(node_config_dict, dict):
                logger.warning(f"The configuration of the {i + 1}th node in the file '{file_path}' is not a dictionary and has been skipped.")
                continue

            node_id = node_config_dict.get("node_id")
            if not node_id:
                raise ValueError(f"The 'node_id' is missing in the configuration of the {i + 1}th node in the file '{file_path}'.")

            # Verify the uniqueness of the node ID
            if node_id in node_ids:
                raise ValueError(f"Duplicate node ID '{node_id}' found in the configuration file '{file_path}'.")
            node_ids.add(node_id)

            if "node_type" not in node_config_dict:
                raise ValueError(f"Node '{node_id}' is missing 'node_type'")
            node_type_str = node_config_dict.get("node_type").upper()
            try:
                node_type = NodeType[node_type_str]
            except KeyError:
                raise ValueError(f"The 'node_type' ('{node_type_str}') of node '{node_id}' in the file '{file_path}' is invalid.")

            node_role_str = node_config_dict.get("node_role", "DEFAULT").upper()
            try:
                node_role = NodeRole[node_role_str]
            except KeyError:
                raise ValueError(f"The 'node_role' ('{node_role_str}') of node '{node_id}' in the file '{file_path}' is invalid.")

            # Whether this node only performs forward computation; defaults to False if not specified
            only_forward_compute = node_config_dict.get("only_forward_compute", False)
            # The agent group to which this node belongs; defaults to 0 if not specified
            agent_group = node_config_dict.get("agent_group", 0)

            dependencies = node_config_dict.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise ValueError(f"The 'dependencies' of node '{node_id}' in the file '{file_path}' must be a list.")

            # Renamed 'config' from node_config_dict to avoid conflict with outer scope 'config' variable
            node_specific_config = resolve_refs(node_config_dict.get("config", {}), global_config)
            executable_ref_str = node_config_dict.get("executable_ref")

            # Add node_id to its own config for easy access within the executable function (e.g., for logging)
            node_specific_config["_node_id_"] = node_id

            # Multi-agent need extra params
            agent_options = node_config_dict.get("agent_options", None)
            
            dag_node = Node(
                node_id=node_id,
                node_type=node_type,
                node_role=node_role,
                only_forward_compute=only_forward_compute,
                agent_group=agent_group,
                dependencies=dependencies,
                config=node_specific_config,  # Use the renamed variable here
                executable_ref=executable_ref_str,
                agent_options=agent_options
            )
            dag_nodes.append(dag_node)
        task_graph = TaskGraph(dag_id)
        task_graph.add_nodes(dag_nodes)
        # Build adjacency lists and validate the graph
        task_graph.build_adjacency_lists()
        valid, msg = task_graph.validate_graph()
        if not valid:
            raise ValueError(f"The graph loaded from the configuration is invalid: {msg}")

        logger.info(f"TaskGraph '{dag_id}' built successfully with {len(task_graph.nodes)} nodes")
        return task_graph

    @staticmethod
    def load_from_file(file_path: str, file_type: str = "yaml") -> TaskGraph:
        """
        Loads and constructs a TaskGraph from the specified YAML or JSON file.
        Determines the file type based on file_type.

        Args:
            file_path (str): The path of the configuration file.
            file_type (str): The type of the configuration file, default is yaml

        Returns:
            TaskGraph: A TaskGraph object constructed from the configuration file.
        """
        raw_dag_config: Optional[Dict[str, Any]] = None  # Initialize to None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_type in ["yaml", "yml"]:
                    raw_dag_config = yaml.safe_load(f)
                elif file_type == "json":
                    raw_dag_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported file type: '{file_type}'. Please use yaml, yml, or json.")
        except FileNotFoundError:
            logger.error(f"The configuration file '{file_path}' was not found.")
            raise
        except yaml.YAMLError as e:  # Specific exception for YAML parsing errors
            logger.error(f"Failed to parse the YAML file '{file_path}': {e}")
            raise
        except json.JSONDecodeError as e:  # Specific exception for JSON parsing errors
            logger.error(f"Failed to parse the JSON file '{file_path}': {e}")
            raise
        except Exception as e:  # Catch-all for other potential loading errors
            logger.error(f"An unknown error occurred while loading the configuration file '{file_path}': {e}")
            raise

        # Check if loading resulted in None (e.g., empty file might be parsed as None by yaml/json libs)
        if raw_dag_config is None:
            raise ValueError(f"The result of loading the configuration file '{file_path}' is empty. It might be an empty file or have a format issue.")

        # Delegate the rest of the parsing to the helper function
        return DAGConfigLoader._parse_raw_config(raw_dag_config, file_path)
