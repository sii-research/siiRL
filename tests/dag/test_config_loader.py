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

import importlib
import json
import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from loguru import logger

from siirl.workers.dag import DAGConfigLoader, Node, NodeRole, NodeType, TaskGraph
from siirl.utils.params import parse_config, ProfilerArguments
from siirl.utils.debug import DistProfiler


@contextmanager
def capture_loguru_logs(level="INFO"):
    """Capture loguru-based logs."""
    logs = []
    handler_id = logger.add(lambda msg: logs.append(msg), level=level)
    yield logs
    logger.remove(handler_id)


# These functions simulate the actual tasks that nodes would execute.
# They are defined as async to match the Node.run signature.
dummy_tasks_content = """
import asyncio

async def load_data_node(node_config):
    return {"data": "loaded_data"}

async def preprocess_data_node(input_data, node_config):
    return {"processed_data": "processed_data"}

async def train_model(input_data, node_config):
    return {"model_weights": "trained_model"}

async def evaluate_ensemble_node(input_data, node_config):
    return {"evaluation_results": "eval_results"}

async def prepare_and_sync_barrier_data(input_data, node_config):
    return {"sync_status": "synced"}

non_callable_object = "I am not callable"

async def missing_param_func():
    return "This function expects no params"
"""


class TestDAGConfigLoader(unittest.TestCase):
    """
    Unit tests for the DAGConfigLoader class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up for all tests: Create a dummy my_tasks.py module.
        """
        cls.dummy_tasks_file = "sii_rl_test_tasks.py"
        with open(cls.dummy_tasks_file, "w", encoding="utf-8") as f:
            f.write(dummy_tasks_content)
        # Add the directory containing the dummy module to sys.path
        sys.path.insert(0, os.path.dirname(os.path.abspath(cls.dummy_tasks_file)))
        # Reload importlib's module cache to ensure it finds the new module
        importlib.invalidate_caches()

    @classmethod
    def tearDownClass(cls):
        """
        Clean up after all tests: Remove the dummy my_tasks.py module.
        """
        if os.path.exists(cls.dummy_tasks_file):
            os.remove(cls.dummy_tasks_file)
        # Remove the dummy module's directory from sys.path
        if sys.path[0] == os.path.dirname(os.path.abspath(cls.dummy_tasks_file)):
            sys.path.pop(0)
        # Clean up any loaded dummy module from sys.modules
        if "sii_rl_test_tasks" in sys.modules:
            del sys.modules["sii_rl_test_tasks"]

    def setUp(self):
        """
        Set up before each test: Initialize DAGConfigLoader.
        """
        self.loader = DAGConfigLoader()
        self.yaml_file = "test_dag.yaml"
        self.json_file = "test_dag.json"

    def tearDown(self):
        """
        Clean up after each test: Remove temporary config files.
        """
        if os.path.exists(self.yaml_file):
            os.remove(self.yaml_file)
        if os.path.exists(self.json_file):
            os.remove(self.json_file)

    def _create_config_file(self, content: str, file_type: str = "yaml"):
        """Helper to create a temporary config file."""
        file_path = self.yaml_file if file_type == "yaml" else self.json_file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def test_load_valid_yaml_config(self):
        """Test loading a valid YAML configuration."""
        yaml_content = """
        dag_id: "test_dag_yaml"
        description: "A valid YAML DAG"
        global_config:
          param1: value1
        nodes:
          - node_id: "start_node"
            node_type: "DATA_LOAD"
            dependencies: []
            executable_ref: "sii_rl_test_tasks.load_data_node"
          - node_id: "middle_node"
            node_type: "COMPUTE"
            dependencies: ["start_node"]
            config:
              compute_param: 100
            executable_ref: "sii_rl_test_tasks.preprocess_data_node"
          - node_id: "end_node"
            node_type: "MODEL_TRAIN"
            dependencies: ["middle_node"]
            node_role: "ACTOR"
            executable_ref: "sii_rl_test_tasks.train_model"
        """
        file_path = self._create_config_file(yaml_content, "yaml")
        task_graph = self.loader.load_from_file(file_path, "yaml")

        self.assertIsInstance(task_graph, TaskGraph)
        self.assertEqual(task_graph.graph_id, "test_dag_yaml")
        self.assertEqual(len(task_graph.nodes), 3)

        start_node = task_graph.get_node("start_node")
        self.assertIsNotNone(start_node)
        self.assertEqual(start_node.node_type, NodeType.DATA_LOAD)
        self.assertEqual(start_node.node_role, NodeRole.DEFAULT)
        self.assertEqual(start_node.dependencies, [])
        self.assertIsNotNone(start_node.executable)
        self.assertEqual(start_node.executable.__name__, "load_data_node")

        middle_node = task_graph.get_node("middle_node")
        self.assertIsNotNone(middle_node)
        self.assertEqual(middle_node.node_type, NodeType.COMPUTE)
        self.assertEqual(middle_node.dependencies, ["start_node"])
        self.assertEqual(middle_node.config, {"compute_param": 100, "_node_id_": "middle_node"})
        self.assertIsNotNone(middle_node.executable)
        self.assertEqual(middle_node.executable.__name__, "preprocess_data_node")

        end_node = task_graph.get_node("end_node")
        self.assertIsNotNone(end_node)
        self.assertEqual(end_node.node_type, NodeType.MODEL_TRAIN)
        self.assertEqual(end_node.node_role, NodeRole.ACTOR)
        self.assertEqual(end_node.dependencies, ["middle_node"])
        self.assertIsNotNone(end_node.executable)
        self.assertEqual(end_node.executable.__name__, "train_model")

        # Test topological sort
        topological_order = task_graph.get_topological_sort()
        self.assertEqual(topological_order, ["start_node", "middle_node", "end_node"])

    def test_load_valid_json_config(self):
        """Test loading a valid JSON configuration."""
        json_content = json.dumps(
            {
                "dag_id": "test_dag_json",
                "description": "A valid JSON DAG",
                "global_config": {"json_param": "json_value"},
                "nodes": [{"node_id": "node_A", "node_type": "COMPUTE", "dependencies": [], "executable_ref": "sii_rl_test_tasks.load_data_node"}, {"node_id": "node_B", "node_type": "COMPUTE", "dependencies": ["node_A"], "executable_ref": "sii_rl_test_tasks.preprocess_data_node"}],
            }
        )
        file_path = self._create_config_file(json_content, "json")
        task_graph = self.loader.load_from_file(file_path, "json")

        self.assertIsInstance(task_graph, TaskGraph)
        self.assertEqual(task_graph.graph_id, "test_dag_json")
        self.assertEqual(len(task_graph.nodes), 2)
        self.assertEqual(task_graph.get_node("node_A").node_type, NodeType.COMPUTE)
        self.assertEqual(task_graph.get_node("node_B").dependencies, ["node_A"])

    def test_missing_dag_id(self):
        """Test error when 'dag_id' is missing."""
        yaml_content = """
        description: "Missing ID"
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            dependencies: []
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "The 'dag_id' is missing in the configuration"):
            self.loader.load_from_file(file_path, "yaml")

    def test_missing_nodes_list(self):
        """Test error when 'nodes' list is missing."""
        yaml_content = """
        dag_id: "test_missing_nodes"
        description: "Missing nodes list"
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "The 'nodes' list is missing in the DAG configuration"):
            self.loader.load_from_file(file_path, "yaml")

    def test_empty_nodes_list(self):
        """Test loading with an empty 'nodes' list (should be valid)."""
        yaml_content = """
        dag_id: "test_empty_nodes"
        nodes: []
        """
        file_path = self._create_config_file(yaml_content)
        task_graph = self.loader.load_from_file(file_path, "yaml")
        self.assertIsInstance(task_graph, TaskGraph)
        self.assertEqual(task_graph.graph_id, "test_empty_nodes")
        self.assertEqual(len(task_graph.nodes), 0)

    def test_missing_node_id(self):
        """Test error when a node is missing 'node_id'."""
        yaml_content = """
        dag_id: "test_missing_node_id"
        nodes:
          - node_type: "COMPUTE"
            dependencies: []
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "The 'node_id' is missing"):
            self.loader.load_from_file(file_path, "yaml")

    def test_missing_node_type(self):
        """Test error when a node is missing 'node_type'."""
        yaml_content = """
        dag_id: "test_missing_node_type"
        nodes:
          - node_id: "node1"
            dependencies: []
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "Node 'node1' is missing 'node_type'"):
            self.loader.load_from_file(file_path, "yaml")

    def test_invalid_node_type(self):
        """Test error with an invalid node type string."""
        yaml_content = """
        dag_id: "test_invalid_node_type"
        nodes:
          - node_id: "node1"
            node_type: "INVALID_TYPE"
            dependencies: []
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "The 'node_type' .* is invalid."):
            self.loader.load_from_file(file_path, "yaml")

    def test_invalid_node_role(self):
        """Test error with an invalid node role string."""
        yaml_content = """
        dag_id: "test_invalid_node_role"
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            node_role: "INVALID_ROLE"
            dependencies: []
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "The 'node_role' .* is invalid."):
            self.loader.load_from_file(file_path, "yaml")

    def test_duplicate_node_ids(self):
        """Test warning for duplicate node IDs (TaskGraph handles replacement)."""
        yaml_content = """
        dag_id: "test_duplicate_nodes"
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            dependencies: []
          - node_id: "node1"
            node_type: "DATA_LOAD"
            dependencies: []
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "Duplicate node ID"):
            task_graph = self.loader.load_from_file(file_path, "yaml")

    def test_non_existent_dependency(self):
        """Test error when a dependency does not exist in the graph."""
        yaml_content = """
        dag_id: "test_non_existent_dep"
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            dependencies: ["non_existent_node"]
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "The dependency 'non_existent_node' of node 'node1' does not exist in the graph."):
            self.loader.load_from_file(file_path, "yaml")

    def test_circular_dependency(self):
        """Test error when a circular dependency is detected."""
        yaml_content = """
        dag_id: "test_circular_dep"
        nodes:
          - node_id: "nodeA"
            node_type: "COMPUTE"
            dependencies: ["nodeB"]
          - node_id: "nodeB"
            node_type: "COMPUTE"
            dependencies: ["nodeC"]
          - node_id: "nodeC"
            node_type: "COMPUTE"
            dependencies: ["nodeA"]
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "There are circular dependencies in graph 'test_circular_dep'"):
            self.loader.load_from_file(file_path, "yaml")

    def test_executable_ref_not_found_module(self):
        """Test error when executable_ref points to a non-existent module."""
        yaml_content = """
        dag_id: "test_invalid_exec_ref_module"
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            executable_ref: "non_existent_module.some_function"
            dependencies: []
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ImportError, "Failed to load the executable function from 'non_existent_module.some_function'"):
            self.loader.load_from_file(file_path, "yaml")

    def test_executable_ref_not_found_function(self):
        """Test error when executable_ref points to a non-existent function in a valid module."""
        yaml_content = f"""
        dag_id: "test_invalid_exec_ref_function"
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            executable_ref: "{self.dummy_tasks_file.replace(".py", "")}.non_existent_function"
            dependencies: []
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ImportError, "Failed to load the executable function from 'sii_rl_test_tasks.non_existent_function'"):
            self.loader.load_from_file(file_path, "yaml")

    def test_executable_ref_not_callable(self):
        """Test error when executable_ref points to an object that is not callable."""
        yaml_content = f"""
        dag_id: "test_exec_ref_not_callable"
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            executable_ref: "{self.dummy_tasks_file.replace(".py", "")}.non_callable_object"
            dependencies: []
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ImportError, "The object resolved from 'sii_rl_test_tasks.non_callable_object' is not callable."):
            self.loader.load_from_file(file_path, "yaml")

    def test_ref_resolution(self):
        """Test correct resolution of !Ref tags."""
        yaml_content = """
        dag_id: "test_ref_resolution"
        global_config:
          default_batch_size: 64
          data_source: "s3://my-bucket/data"
          nested:
            level1:
              level2: "deep_value"
        nodes:
          - node_id: "node1"
            node_type: "DATA_LOAD"
            dependencies: []
            config:
              batch_size: !Ref global_config.default_batch_size
              source: !Ref global_config.data_source
              deep_param: !Ref global_config.nested.level1.level2
            executable_ref: "sii_rl_test_tasks.load_data_node"
        """
        file_path = self._create_config_file(yaml_content)
        task_graph = self.loader.load_from_file(file_path, "yaml")
        node1 = task_graph.get_node("node1")
        self.assertEqual(node1.config["batch_size"], 64)
        self.assertEqual(node1.config["source"], "s3://my-bucket/data")
        self.assertEqual(node1.config["deep_param"], "deep_value")

    def test_ref_resolution_invalid_path(self):
        """Test error when !Ref points to an invalid path."""
        yaml_content = """
        dag_id: "test_ref_invalid_path"
        global_config:
          param1: value1
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            dependencies: []
            config:
              invalid_ref: !Ref global_config.non_existent_key
            executable_ref: "sii_rl_test_tasks.load_data_node"
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "Unresolved reference 'global_config.non_existent_key'."):
            self.loader.load_from_file(file_path, "yaml")

    def test_ref_resolution_invalid_path_nested(self):
        """Test error when !Ref points to an invalid nested path."""
        yaml_content = """
        dag_id: "test_ref_invalid_nested_path"
        global_config:
          param1:
            sub_param: value1
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            dependencies: []
            config:
              invalid_ref: !Ref global_config.param1.non_existent_sub_key
            executable_ref: "sii_rl_test_tasks.load_data_node"
        """
        file_path = self._create_config_file(yaml_content)
        with self.assertRaisesRegex(ValueError, "Unresolved reference 'global_config.param1.non_existent_sub_key'."):
            self.loader.load_from_file(file_path, "yaml")

    def test_node_role_validation(self):
        """Test node role validation for non-model node types."""
        # This should pass: COMPUTE node with DEFAULT role
        yaml_content_valid = """
        dag_id: "test_role_valid"
        nodes:
          - node_id: "node1"
            node_type: "COMPUTE"
            node_role: "DEFAULT"
            dependencies: []
        """
        file_path_valid = self._create_config_file(yaml_content_valid)
        task_graph_valid = self.loader.load_from_file(file_path_valid, "yaml")
        self.assertEqual(task_graph_valid.get_node("node1").node_role, NodeRole.DEFAULT)

        # This should fail: DATA_LOAD node with ACTOR role
        yaml_content_invalid = """
        dag_id: "test_role_invalid"
        nodes:
          - node_id: "node2"
            node_type: "DATA_LOAD"
            node_role: "ACTOR"
            dependencies: []
        """
        file_path_invalid = self._create_config_file(yaml_content_invalid)
        with self.assertRaisesRegex(ValueError, "The role type of non-model nodes must be DEFAULT"):
            self.loader.load_from_file(file_path_invalid, "yaml")

        # This should pass: MODEL_TRAIN node with ACTOR role
        yaml_content_model_actor = """
        dag_id: "test_model_actor"
        nodes:
          - node_id: "node3"
            node_type: "MODEL_TRAIN"
            node_role: "ACTOR"
            dependencies: []
        """
        file_path_model_actor = self._create_config_file(yaml_content_model_actor)
        task_graph_model_actor = self.loader.load_from_file(file_path_model_actor, "yaml")
        self.assertEqual(task_graph_model_actor.get_node("node3").node_role, NodeRole.ACTOR)

    @patch("graphviz.Digraph")
    def test_save_dag_pic_no_nodes(self, mock_digraph):
        """Test save_dag_pic with no nodes in the graph."""
        yaml_content = """
        dag_id: "test_empty_graph_pic"
        nodes: []
        """
        file_path = self._create_config_file(yaml_content)
        task_graph = self.loader.load_from_file(file_path, "yaml")
        with capture_loguru_logs(level="WARNING") as logs:
            result = task_graph.save_dag_pic()
            self.assertIn("DAG graph 'test_empty_graph_pic' is empty. No image will be generated.", "\n".join(logs))
        self.assertIsNone(result)
        mock_digraph.assert_not_called()

    @patch("graphviz.Digraph")
    def test_save_dag_pic_invalid_graph(self, mock_digraph):
        """Test save_dag_pic with an invalid graph (circular dependency)."""
        yaml_content = """
        dag_id: "test_circular_dep_pic"
        nodes:
          - node_id: "nodeA"
            node_type: "COMPUTE"
            dependencies: ["nodeB"]
          - node_id: "nodeB"
            node_type: "COMPUTE"
            dependencies: ["nodeA"]
        """
        file_path = self._create_config_file(yaml_content)
        # We expect load_config to raise ValueError for circular dependency
        with self.assertRaises(ValueError):
            self.loader.load_from_file(file_path, "yaml")

        # To test save_dag_pic on an invalid graph, we'd need to bypass load_config's validation
        # or create an invalid graph manually. Let's create one manually.
        graph = TaskGraph("manual_invalid_graph")
        nodeA = Node("nodeA", NodeType.COMPUTE, dependencies=["nodeB"])
        nodeB = Node("nodeB", NodeType.COMPUTE, dependencies=["nodeA"])
        graph.add_node(nodeA)
        graph.add_node(nodeB)
        graph.build_adjacency_lists()  # Ensure adj lists are built for validation

        with capture_loguru_logs(level="ERROR") as logs:
            result = graph.save_dag_pic()
            self.assertIn("Graph 'manual_invalid_graph' is invalid. Unable to generate image:", "\n".join(logs))
        self.assertIsNone(result)
        mock_digraph.assert_not_called()
    
    def test_load_profiler_yaml_config(self):
        """Test loading a valid profiler configuration."""
        yaml_context="""
        data: null
        actor_rollout_ref: null
        critic: null
        reward_model: null
        custom_reward_function: null
        algorithm: null
        trainer: null
        dag: null
        profiler:
          enable: True
          save_path: './prof_data'
          level: 'level1'
          with_memory: False
          record_shapes: False
          with_npu: True
          with_cpu: False
          with_module: False
          with_stack: False
          analysis: True
          discrete: False
          roles: ['generate', 'compute_reward']
          all_ranks: False
          ranks: [0]
          profile_steps: [0]
        """
        file_path = self._create_config_file(yaml_context, "yaml")
        from omegaconf import OmegaConf
        yaml_dict = OmegaConf.load(file_path)
        profiler = parse_config(yaml_dict).profiler
        self.assertIsInstance(profiler, ProfilerArguments)
        self.assertTrue(profiler.enable)
        self.assertEqual(profiler.level, "level1")
        self.assertFalse(profiler.with_memory)
        self.assertTrue(profiler.with_npu)

        def is_subset(subset, superset):
            set_subset = set(subset)
            set_superset = set(superset)
            return set_subset.issubset(set_superset)
        self.assertTrue(is_subset(profiler.roles, ["generate", "compute_reward", "compute_old_log_prob", 
                        "compute_ref_log_porb", "compute_value", "compute_advantage", "train_critic", "train_actor"]))

    def test_profiler_npu_environment(self):
        """Test npu environment for profiler."""
        config = ProfilerArguments(enable=True)
        profiler = DistProfiler(rank=0, config=config)
        from siirl.utils.extras.device import is_npu_available
        if not is_npu_available:
            self.assertFalse(profiler.config.enable)
        else:
            self.assertTrue(profiler.config.enable)



if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
