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

from typing import Dict, Tuple, Type, Any, Optional
from dataclasses import asdict, is_dataclass

from dacite import Config as DaciteConfig, from_dict
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from siirl.workers.dag import NodeRole, NodeType, TaskGraph
from siirl.utils.params import ActorRolloutRefArguments, CriticArguments, RewardModelArguments, SiiRLArguments

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from siirl.workers.dag import TaskGraph
from siirl.utils.params import log_dict_formatted

NODE_ID = "_node_id_"
INTERN_CONFIG = "intern_config"


def unflatten_dict_with_omegaconf(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflattens a flat dictionary with dot-separated keys into a nested dictionary using OmegaConf.

    Args:
        flat_dict: A dictionary where keys might be dot-separated (e.g., 'model.name').

    Returns:
        A nested dictionary.
    """
    if not flat_dict:
        return {}
    config = OmegaConf.create()
    for key, value in flat_dict.items():
        try:
            OmegaConf.update(config, key, value, merge=True, force_add=True)
        except Exception as e:
            logger.error(f"OmegaConf.update failed for key='{key}', value='{value}': {e}")
            raise
    return OmegaConf.to_container(config, resolve=True, throw_on_missing=False)


def update_task_graph_node_configs(workerflow_taskgraph: TaskGraph, basic_common_config: "SiiRLArguments") -> TaskGraph:
    """
    Updates node configurations by merging global defaults with node-specific overrides,
    and stores the resulting configuration as both a dictionary and a dataclass instance.

    Args:
        workerflow_taskgraph: The TaskGraph whose nodes will be updated.
        basic_common_config: The global SiiRLArguments with default settings.

    Returns:
        The updated TaskGraph.
    """
    logger.info("Starting update of TaskGraph node configurations (using OmegaConf and Dacite)...")
    workerflow_taskgraph.build_adjacency_lists()

    node_role_config_map: Dict[NodeRole, Tuple[str, Type]] = {
        NodeRole.ACTOR: ("actor_rollout_ref", ActorRolloutRefArguments),
        NodeRole.ROLLOUT: ("actor_rollout_ref", ActorRolloutRefArguments),
        NodeRole.REFERENCE: ("actor_rollout_ref", ActorRolloutRefArguments),
        NodeRole.CRITIC: ("critic", CriticArguments),
        NodeRole.REWARD: ("reward_model", RewardModelArguments),
    }

    for node_id, node in workerflow_taskgraph.nodes.items():
        if node.node_type not in [NodeType.MODEL_INFERENCE, NodeType.MODEL_TRAIN]:
            logger.debug(f"Node '{node.node_id}' of type {node.node_type} skipped for config update.")
            continue

        original_node_config_flat = node.config or {}
        original_node_config_dict = unflatten_dict_with_omegaconf(original_node_config_flat)

        if NODE_ID in original_node_config_dict:
            del original_node_config_dict[NODE_ID]
        node_specific_omega_conf = OmegaConf.create(original_node_config_dict)

        if node.node_role in node_role_config_map:
            default_config_attr_name, target_dataclass_type = node_role_config_map[node.node_role]
            default_config_branch_instance = getattr(basic_common_config, default_config_attr_name, None)

            merged_omega_conf: Optional[DictConfig] = None

            if default_config_branch_instance is None:
                logger.warning(f"Global default config '{default_config_attr_name}' not in basic_common_config for node '{node.node_id}'. Using only node-specific config.")
                merged_omega_conf = node_specific_omega_conf
            else:
                default_config_branch_dict = asdict(default_config_branch_instance)
                default_config_branch_omega_base = OmegaConf.create(default_config_branch_dict)

                if not isinstance(default_config_branch_omega_base, DictConfig):
                    logger.error(f"Global config for '{default_config_attr_name}' is not a DictConfig. Cannot merge. Using only node-specific config for node '{node.node_id}'.")
                    merged_omega_conf = node_specific_omega_conf
                else:
                    merged_omega_conf = OmegaConf.merge(default_config_branch_omega_base.copy(), node_specific_omega_conf)

            merged_config_dict = OmegaConf.to_container(merged_omega_conf, resolve=True, throw_on_missing=False)
            if not isinstance(merged_config_dict, dict):
                raise ValueError(f"Merged config for node '{node.node_id}' is not a dictionary.")

            try:
                # Convert the merged dictionary back into a validated dataclass instance
                merged_dataclass_instance = from_dict(data_class=target_dataclass_type, data=merged_config_dict, config=DaciteConfig(check_types=False))
            except Exception as e:
                logger.error(f"Dacite conversion to '{target_dataclass_type.__name__}' failed for node '{node.node_id}': {e}")
                raise
            node.config = {INTERN_CONFIG: merged_dataclass_instance, NODE_ID: node.node_id}

        else:
            logger.warning(f"Node '{node.node_id}' ({node.node_role}) has an unmapped role. Using its unflattened original configuration without creating a dataclass instance.")
            node.config = original_node_config_dict

    logger.info("TaskGraph node configuration update complete.")
    return workerflow_taskgraph


def display_node_config(workerflow_taskgraph: TaskGraph) -> None:
    """
    Prints the configuration for each node.
    This version is adapted for when node.config primarily holds a dataclass instance.
    """
    if not isinstance(workerflow_taskgraph, TaskGraph):
        logger.error("Error: Input must be a TaskGraph object.")
        return

    if not workerflow_taskgraph.nodes:
        logger.warning(f"Graph '{workerflow_taskgraph.graph_id}' has no nodes.")
        return

    logger.debug(f"Displaying configurations for all nodes in graph '{workerflow_taskgraph.graph_id}':")

    for node_id, node in workerflow_taskgraph.nodes.items():
        if not isinstance(node.config, dict):
            logger.warning(f"Node '{node_id}' config is not a dictionary. Skipping.")
            continue

        dataclass_obj = node.config.get(INTERN_CONFIG)

        if dataclass_obj and is_dataclass(dataclass_obj):
            config_for_display = asdict(dataclass_obj)

            # Include the node ID in the displayed configuration
            if NODE_ID in node.config:
                config_for_display[NODE_ID] = node.config[NODE_ID]

            log_dict_formatted(config_for_display, title=f"Node: {node_id} Configuration Details", log_level="debug")
        else:
            # If the config is not a dataclass, log the raw dictionary
            logger.warning(f"Node '{node_id}' does not contain a valid dataclass in '{INTERN_CONFIG}'.")
            log_dict_formatted(node.config, title=f"Node: {node_id} Raw Configuration Details", log_level="debug")
