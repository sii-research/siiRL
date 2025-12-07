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

"""
Simplified Pipeline Builder API for defining DAG workflows in Python code.

This module provides a clean, intuitive API for users to define their training
pipelines directly in Python, with explicit function bindings for each node.
"""

from typing import Callable, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from loguru import logger

from siirl.execution.dag.node import Node, NodeType
from siirl.execution.dag.task_graph import TaskGraph


@dataclass
class NodeConfig:
    """Configuration for a pipeline node."""
    agent_group: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """
    Simplified Pipeline builder for defining training workflows in Python.

    Users can directly specify the function to execute for each node,
    making the entire workflow transparent and easy to understand.

    Example:
        >>> pipeline = Pipeline("grpo_training")
        >>> pipeline.add_node(
        ...     "rollout",
        ...     func="siirl.dag_worker.dagworker:DAGWorker.generate",
        ...     deps=[]
        ... )
        >>> pipeline.add_node(
        ...     "reward",
        ...     func="siirl.dag_worker.dagworker:DAGWorker.compute_reward",
        ...     deps=["rollout"]
        ... )
        >>> graph = pipeline.build()
    """

    def __init__(self, pipeline_id: str, description: str = ""):
        """
        Initialize a Pipeline builder.

        Args:
            pipeline_id: Unique identifier for this pipeline
            description: Human-readable description of the pipeline
        """
        self.pipeline_id = pipeline_id
        self.description = description
        self._nodes: Dict[str, Dict[str, Any]] = {}

    def add_node(
        self,
        node_id: str,
        func: Union[str, Callable],
        deps: Optional[List[str]] = None,
        config: Optional[NodeConfig] = None,
        **kwargs
    ) -> "Pipeline":
        """
        Add a node to the pipeline.

        Args:
            node_id: Unique identifier for this node
            func: Function to execute. Can be:
                  - String path: "module.path:ClassName.method" or "module.path:function"
                  - Callable: Direct function reference
            deps: List of node IDs that this node depends on
            config: Node configuration (optional)
            **kwargs: Additional node parameters (e.g., only_forward_compute)

        Returns:
            self: For method chaining

        Raises:
            ValueError: If node_id already exists in the pipeline
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists in pipeline '{self.pipeline_id}'")

        deps = deps or []
        config = config or NodeConfig()

        self._nodes[node_id] = {
            "func": func,
            "deps": deps,
            "config": config,
            "kwargs": kwargs
        }

        logger.debug(f"Added node '{node_id}' to pipeline '{self.pipeline_id}'")
        return self

    def build(self) -> TaskGraph:
        """
        Build and validate the TaskGraph from the pipeline definition.

        Returns:
            TaskGraph: A validated TaskGraph ready for execution

        Raises:
            ValueError: If the pipeline is invalid (e.g., circular dependencies)
        """
        from siirl.execution.dag.node import NodeRole

        task_graph = TaskGraph(graph_id=self.pipeline_id)

        for node_id, node_info in self._nodes.items():
            # Extract node_type and node_role from kwargs if provided, otherwise use defaults
            kwargs = node_info["kwargs"].copy()
            node_type = kwargs.pop("node_type", NodeType.COMPUTE)
            node_role = kwargs.pop("node_role", NodeRole.DEFAULT)

            # Create Node instance
            node = Node(
                node_id=node_id,
                node_type=node_type,
                node_role=node_role,
                dependencies=node_info["deps"],
                agent_group=node_info["config"].agent_group,
                config=node_info["config"].config,
                **kwargs
            )

            # Bind the executable function
            func = node_info["func"]
            if isinstance(func, str):
                # Function specified as string path
                node.executable_ref = func
                node._resolve_executable()
            else:
                # Direct callable
                node.executable = func

            task_graph.add_node(node)

        # Build adjacency lists and validate
        task_graph.build_adjacency_lists()
        valid, msg = task_graph.validate_graph()
        if not valid:
            raise ValueError(f"Invalid pipeline '{self.pipeline_id}': {msg}")

        logger.info(f"Pipeline '{self.pipeline_id}' built successfully with {len(self._nodes)} nodes")
        return task_graph

    def visualize(self, output_path: str = None, directory: str = "./"):
        """
        Visualize the pipeline structure.

        Args:
            output_path: Filename for the visualization (without extension)
            directory: Directory to save the visualization

        Returns:
            TaskGraph: The built task graph
        """
        graph = self.build()
        if output_path:
            graph.save_dag_pic(filename=output_path, directory=directory)
            logger.info(f"Pipeline visualization saved to {directory}/{output_path}")
        return graph
