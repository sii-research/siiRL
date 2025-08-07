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

import importlib
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from siirl.utils.params import log_dict_formatted


class NodeType(Enum):
    """
    Define the types of nodes in the DAG.
    """

    COMPUTE = "COMPUTE"  # General computing task
    DATA_LOAD = "DATA_LOAD"  # Load data from DataLoader
    ENV_INTERACT = "ENV_INTERACT"  # Interact with the environment
    MODEL_INFERENCE = "MODEL_INFERENCE"  # Model inference
    MODEL_TRAIN = "MODEL_TRAIN"  # Model training
    PUT_TO_BUFFER = "PUT_TO_BUFFER"  # Put data into the distributed buffer
    GET_FROM_BUFFER = "GET_FROM_BUFFER"  # Get data from the distributed buffer
    BARRIER_SYNC = "BARRIER_SYNC"  # Global synchronization point
    CUSTOM = "CUSTOM"  # User-defined node type, executed using an executable


class NodeRole(Enum):
    """
    Define the roles that a node plays in a distributed reinforcement learning framework.
    This helps with specific scheduling or resource allocation.
    """

    DEFAULT = "DEFAULT"  # Default
    ACTOR = "ACTOR"  # Actor
    ADVANTAGE = "ADVANTAGE"  # ADVANTAGE
    CRITIC = "CRITIC"  # Critic
    ROLLOUT = "ROLLOUT"  # Rollout
    REFERENCE = "REFERENCE"  # Reference
    REWARD = "REWARD"  # Reward

    POSTPROCESS_SAMPLING = "POSTPROCESS_SAMPLING"  # Post-process sampling for dapo


class NodeStatus(Enum):
    """
    Define the execution status of a DAG node.
    """

    PENDING = "PENDING"  # Waiting for dependencies to complete
    READY = "READY"  # Dependencies completed, ready to execute
    RUNNING = "RUNNING"  # Currently executing
    COMPLETED = "COMPLETED"  # Execution completed successfully
    FAILED = "FAILED"  # Execution failed
    SKIPPED = "SKIPPED"  # Skipped


class Node:
    """
    Represents a node (task unit) in the DAG.
    """

    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        node_role: NodeRole = NodeRole.DEFAULT,
        only_forward_compute: bool = False,
        agent_group: int = 0,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        executable_ref: Optional[str] = None,
        user_options: Optional[str] = {},
        retry_limit: int = 0,
    ):
        """
        Initialize a node.

        Args:
            node_id (str): The unique identifier of the node.
            node_type (NodeType): The type of the node.
            node_role (NodeRole): The role played by the node. Defaults to NodeRole.DEFAULT.
            dependencies (Optional[List[str]]): A list of IDs of other nodes that this node depends on. Defaults to an empty list.
            config (Optional[Dict[str, Any]]): Specific configuration information for the node. Defaults to an empty dictionary.
            executable_ref (Optional[str]): A string reference to the Python function for the node's execution logic
                                           (e.g., "my_module.my_submodule.my_function").
                                           If None, it means the node may have built-in logic or be handled by an external executor.
            retry_limit (int): The maximum number of retries when the node execution fails. Defaults to 0 (no retries).
        """
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a non-empty string.")
        if not isinstance(node_type, NodeType):
            raise ValueError("node_type must be a member of the NodeType enum.")
        if not isinstance(node_role, NodeRole):
            raise ValueError("node_role must be a member of the NodeRole enum.")
        if node_type not in [NodeType.COMPUTE, NodeType.MODEL_TRAIN, NodeType.MODEL_INFERENCE] and node_role != NodeRole.DEFAULT:
            raise ValueError("The role type of non-model nodes must be DEFAULT")

        self.node_id: str = node_id
        self.node_type: NodeType = node_type
        self.node_role: NodeRole = node_role
        self.only_forward_compute: bool = only_forward_compute
        self.agent_group: int = agent_group
        self.dependencies: List[str] = dependencies or []
        self.config: Dict[str, Any] = config or {}
        self.executable_ref: Optional[str] = executable_ref
        self.retry_limit: int = retry_limit
        self.retries_done: int = 0

        self.async_rollout = None
        self.mode = "sync"
        self._executable: Optional[Callable] = None
        self.output: Any = None  # Store the result of the node execution
        self.error_info: Optional[str] = None  # Store error information when the node fails
        self.user_options: Dict = user_options
        self.config["user_options"] = self.user_options
        if self.executable_ref:
            self._resolve_executable()

        self.status: NodeStatus = NodeStatus.PENDING

    def _resolve_executable(self) -> None:
        """
        Dynamically import and obtain the executable function based on the executable_ref string.
        """
        if not self.executable_ref:
            self._executable = None
            return

        try:
            module_path, function_name = self.executable_ref.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self._executable = getattr(module, function_name)
            if not callable(self._executable):
                raise AttributeError(f"The object resolved from '{self.executable_ref}' is not callable.")
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(f"Failed to load the executable function from '{self.executable_ref}': {e}") from e

    @property
    def executable(self) -> Optional[Callable]:
        """Return the resolved executable function."""
        return self._executable

    @executable.setter
    def executable(self, execute: Optional[Callable]):
        """Set the executable function for this node."""
        self._executable = execute

    def add_dependency(self, dependency_id: str) -> None:
        """
        Add a dependency.
        Args:
            dependency_id (str): The ID of the dependent node.
        """
        if dependency_id not in self.dependencies:
            self.dependencies.append(dependency_id)

    def remove_dependency(self, dependency_id: str) -> None:
        """
        Remove a dependency.
        Args:
            dependency_id (str): The ID of the dependency node to be removed.
        """
        if dependency_id in self.dependencies:
            self.dependencies.remove(dependency_id)

    def is_ready(self, completed_node_ids: Set[str]) -> bool:
        """
        Check if all dependencies of this node have been completed.
        Args:
            completed_node_ids (Set[str]): A set of IDs of completed nodes.
        Returns:
            bool: True if all dependencies are completed, otherwise False.
        """
        if self.status != NodeStatus.PENDING:  # Only nodes in PENDING status can become READY
            return False
        return all(dep_id in completed_node_ids for dep_id in self.dependencies)

    def update_status(self, new_status: NodeStatus, error_info: Optional[str] = None) -> None:
        """Update the node status and record error information (if applicable)."""
        self.status = new_status
        if error_info:
            self.error_info = error_info
        if new_status == NodeStatus.FAILED:
            logger.error(f"Node {self.node_id} execution failed: {error_info or 'Unknown error'}")
        elif new_status == NodeStatus.COMPLETED:
            self.error_info = None  # Clear previous error information

    def update_config(self, new_config_items: Dict[str, Any], overwrite: bool = True) -> None:
        """
        Update the node's configuration.

        Args:
            new_config_items (Dict[str, Any]): A dictionary containing configuration keys and values to add or update.
            overwrite (bool): If True (default), existing keys in the node's config will be overwritten
                              by those in new_config_items. If False, existing keys will be preserved,
                              and only new keys from new_config_items will be added.
        """
        if not isinstance(new_config_items, dict):
            logger.warning(f"Node {self.node_id}: Failed to update config. Provided new_config_items is not a dictionary (type: {type(new_config_items)}).")
            return

        if overwrite:
            self.config.update(new_config_items)
            logger.info(f"Node {self.node_id}: Configuration updated (overwrite=True).")
        else:
            for key, value in new_config_items.items():
                if key not in self.config:
                    self.config[key] = value
            logger.info(f"Node {self.node_id}: Configuration updated (overwrite=False, existing keys preserved).")

        log_dict_formatted(self.config, title=f"Node {self.node_id} current config", log_level="debug")

    def can_retry(self) -> bool:
        """Check if the node can be retried."""
        return self.status == NodeStatus.FAILED and self.retries_done < self.retry_limit

    def increment_retry_count(self) -> None:
        """Increment the retry count."""
        self.retries_done += 1

    def run(self, **kwargs: Any) -> Any:
        """
        Execute the task of the node.
        Args:
            **kwargs: Parameters passed to the executable function, usually the outputs of its dependent nodes.
        Returns:
            Any: The result of the node execution.
        """
        logger.debug(f"Starting to execute node: {self.node_id} (Type: {self.node_type.value}, Role: {self.node_role.value})")
        self.update_status(NodeStatus.RUNNING)

        if not self.executable:
            # For nodes without an executable reference, they may be handled by an external system,
            # or they are purely structural nodes (e.g., BARRIER_SYNC, whose logic is in the scheduler).
            # one implement for barrier...
            if self.node_type == NodeType.BARRIER_SYNC and kwargs.get("do_barrier", False):
                import torch.distributed as dist

                logger.debug(f"Node {self.node_id} block before barrier ...")
                dist.barrier(group=kwargs.get("barrier_group", None))

            logger.debug(f"Node {self.node_id} has no executable function, skipping execution.")
            self.output = None  # Or set a specific output based on the node type
            return self.output

        try:
            # Prepare the parameters to be passed to the executable function
            # Here, the actual parameters can be constructed based on kwargs and node configuration
            # For example, only pass the outputs of dependencies
            # resolved_args = {dep_id: kwargs.get(dep_id) for dep_id in self.dependencies if dep_id in kwargs}
            # resolved_args.update(self.config) # Or pass the node configuration as well

            # Simplification: Pass all kwargs directly, and the user function handles them
            node_output = self._executable(**kwargs, node_config=self.config)
            self.output = node_output
            self.update_status(NodeStatus.COMPLETED)
            logger.debug(f"Node {self.node_id} execution completed.")
            return self.output
        except Exception as e:
            error_message = f"An error occurred while executing node {self.node_id}: {e}"
            self.update_status(NodeStatus.FAILED, error_message)
            # An exception can be raised here, or the scheduler can handle the FAILED status
            raise RuntimeError(error_message) from e

    def __repr__(self) -> str:
        return f"Node(node_id='{self.node_id}', type='{self.node_type.value}', role='{self.node_role.value}', agent_group='{self.agent_group}', only_forward_compute='{self.only_forward_compute}', status='{self.status.value}', deps={len(self.dependencies)})"

    def copy(self) -> "Node":
        new_node = Node(
            node_id=self.node_id,
            node_type=self.node_type,
            node_role=self.node_role,
            dependencies=list(self.dependencies),
            config=dict(self.config),
            executable_ref=self.executable_ref,
            retry_limit=self.retry_limit,
            only_forward_compute=self.only_forward_compute,
            agent_group=self.agent_group,
            user_options=self.user_options,
        )
        new_node.status = self.status
        new_node.retries_done = self.retries_done
        new_node._executable = self._executable
        return new_node
