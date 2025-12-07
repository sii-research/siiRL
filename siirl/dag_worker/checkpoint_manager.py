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

"""Checkpoint save/load operations for distributed training."""

import os
import torch
import torch.distributed as dist
from typing import Dict, Optional, Any
from loguru import logger

from siirl.execution.dag.node import NodeRole, NodeType
from siirl.params import SiiRLArguments
from siirl.dag_worker.constants import DAGConstants
from siirl.dag_worker.dag_utils import generate_node_worker_key
from siirl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


class CheckpointManager:
    """Manages distributed checkpoint save/load with atomic commits."""

    def __init__(
        self,
        config: SiiRLArguments,
        rank: int,
        gather_group: dist.ProcessGroup,
        workers: Dict[str, Any],
        taskgraph: Any,
        dataloader: Any,
        first_rollout_node: Any,
        get_node_dp_info_fn: callable
    ):
        self.config = config
        self.rank = rank
        self.gather_group = gather_group
        self.workers = workers
        self.taskgraph = taskgraph
        self.dataloader = dataloader
        self.first_rollout_node = first_rollout_node
        self._get_node_dp_info = get_node_dp_info_fn

    def save_checkpoint(self, global_steps: int) -> None:
        """Save checkpoint atomically across all ranks."""
        step_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{global_steps}")
        os.makedirs(step_dir, exist_ok=True)
        dist.barrier(self.gather_group)

        logger.info(f"Rank {self.rank}: Saving checkpoint for global_step {global_steps} to {step_dir}")

        self._save_model_states(global_steps, step_dir)
        self._save_dataloader_state(step_dir)

        logger.debug(f"Rank {self.rank}: All data saved. Waiting at barrier before committing checkpoint.")
        dist.barrier(self.gather_group)

        if self.rank == 0:
            self._commit_checkpoint(global_steps)

        dist.barrier(self.gather_group)
        logger.info(f"Rank {self.rank}: Finished saving and committing checkpoint for step {global_steps}.")

    def _save_model_states(self, global_steps: int, step_dir: str) -> None:
        """Save model states for all trainable nodes."""
        saved_worker_keys = set()

        for node in self.taskgraph.nodes.values():
            if node.node_type != NodeType.MODEL_TRAIN:
                continue
            if node.node_role not in [NodeRole.ACTOR, NodeRole.CRITIC]:
                continue

            node_worker_key = generate_node_worker_key(node)

            if node_worker_key in saved_worker_keys:
                continue

            worker = self.workers[node_worker_key]

            sub_dir_name = f"{node.node_role.name.lower()}_agent_{node.agent_group}"
            checkpoint_path = os.path.join(step_dir, sub_dir_name)

            role_name_for_config = node.node_role.name.lower()
            max_ckpt_keep = getattr(
                self.config.trainer,
                f"max_{role_name_for_config}_ckpt_to_keep",
                10
            )

            worker.save_checkpoint(
                local_path=checkpoint_path,
                global_step=global_steps,
                max_ckpt_to_keep=max_ckpt_keep
            )
            saved_worker_keys.add(node_worker_key)
            logger.debug(
                f"Rank {self.rank}: Saved {node.node_role.name} checkpoint for agent {node.agent_group} "
                f"to {checkpoint_path}"
            )

    def _save_dataloader_state(self, step_dir: str) -> None:
        """Save dataloader state (only TP rank 0 and PP rank 0 per DP group)."""
        _, dp_rank, tp_rank, _, pp_rank, _ = self._get_node_dp_info(self.first_rollout_node)

        if tp_rank == 0 and pp_rank == 0:
            dataloader_path = os.path.join(step_dir, f"data_dp_rank_{dp_rank}.pt")
            dataloader_state = self.dataloader.state_dict()
            torch.save(dataloader_state, dataloader_path)
            logger.debug(
                f"Rank {self.rank} (DP_Rank {dp_rank}, TP_Rank {tp_rank}, PP_Rank {pp_rank}): "
                f"Saved dataloader state to {dataloader_path}"
            )

    def _commit_checkpoint(self, global_steps: int) -> None:
        """Atomically commit checkpoint by writing tracker file (rank 0 only)."""
        tracker_file = os.path.join(
            self.config.trainer.default_local_dir,
            "latest_checkpointed_iteration.txt"
        )
        with open(tracker_file, "w") as f:
            f.write(str(global_steps))
        logger.info(f"Rank 0: Checkpoint for step {global_steps} successfully committed.")

    def load_checkpoint(self) -> int:
        """Load checkpoint and return global step to resume from."""
        if self.config.trainer.resume_mode == "disable":
            if self.rank == 0:
                logger.info("Checkpoint loading is disabled. Starting from scratch.")
            return 0

        checkpoint_path = self._determine_checkpoint_path()

        checkpoint_path_container = [checkpoint_path]
        dist.broadcast_object_list(checkpoint_path_container, src=0)
        global_step_folder = checkpoint_path_container[0]

        if global_step_folder is None:
            if self.rank == 0:
                logger.info("No valid checkpoint to load. Training will start from step 0.")
            dist.barrier(self.gather_group)
            return 0

        try:
            global_steps = int(os.path.basename(global_step_folder).split("global_step_")[-1])
            logger.info(
                f"Rank {self.rank}: Resuming from checkpoint. "
                f"Setting global_steps to {global_steps}."
            )
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Could not parse global step from checkpoint path: {global_step_folder}"
            ) from e

        self._load_model_states(global_step_folder)
        self._load_dataloader_state(global_step_folder)

        dist.barrier(self.gather_group)
        logger.info(f"Rank {self.rank}: Finished loading all checkpoint components.")

        return global_steps

    def _determine_checkpoint_path(self) -> Optional[str]:
        """Determine checkpoint path (rank 0 only)."""
        if self.rank != 0:
            return None

        checkpoint_dir = self.config.trainer.default_local_dir
        resume_from_path = self.config.trainer.resume_from_path
        path_to_load = None

        if self.config.trainer.resume_mode == "auto":
            latest_path = find_latest_ckpt_path(checkpoint_dir)
            if latest_path:
                logger.info(f"Rank 0: Auto-found latest checkpoint at {latest_path}")
                path_to_load = latest_path
        elif self.config.trainer.resume_mode == "resume_path" and resume_from_path:
            logger.info(f"Rank 0: Attempting to load from specified path: {resume_from_path}")
            path_to_load = resume_from_path

        if path_to_load and os.path.exists(path_to_load):
            return path_to_load
        else:
            logger.warning(
                f"Rank 0: Checkpoint path not found or invalid: '{path_to_load}'. "
                f"Starting from scratch."
            )
            return None

    def _load_model_states(self, global_step_folder: str) -> None:
        """Load model states for all trainable nodes."""
        loaded_worker_keys = set()

        for node in self.taskgraph.nodes.values():
            if node.node_type != NodeType.MODEL_TRAIN:
                continue
            if node.node_role not in [NodeRole.ACTOR, NodeRole.CRITIC]:
                continue

            node_worker_key = generate_node_worker_key(node)

            if node_worker_key in loaded_worker_keys:
                continue

            worker = self.workers[node_worker_key]

            sub_dir_name = f"{node.node_role.name.lower()}_agent_{node.agent_group}"
            checkpoint_path = os.path.join(global_step_folder, sub_dir_name)

            if os.path.exists(checkpoint_path):
                worker.load_checkpoint(
                    local_path=checkpoint_path,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load
                )
                loaded_worker_keys.add(node_worker_key)
                logger.debug(
                    f"Rank {self.rank}: Loaded {node.node_role.name} checkpoint for agent "
                    f"{node.agent_group} from {checkpoint_path}"
                )
            else:
                logger.warning(
                    f"Rank {self.rank}: Checkpoint for agent {node.agent_group}'s "
                    f"{node.node_role.name} not found at {checkpoint_path}. "
                    f"Weights will be from initialization. "
                    f"If has multi-agent, will share the same checkpoint in agents"
                )

    def _load_dataloader_state(self, global_step_folder: str) -> None:
        """Load dataloader state for current DP group."""
        _, dp_rank, _, _, _, _ = self._get_node_dp_info(self.first_rollout_node)
        dataloader_path = os.path.join(global_step_folder, f"data_dp_rank_{dp_rank}.pt")

        if os.path.exists(dataloader_path):
            dataloader_state = torch.load(dataloader_path, map_location="cpu")
            self.dataloader.load_state_dict(dataloader_state)
            logger.debug(
                f"Rank {self.rank} (DP_Rank {dp_rank}): Loaded dataloader state from "
                f"{dataloader_path}"
            )
        else:
            logger.warning(
                f"Rank {self.rank} (DP_Rank {dp_rank}): Dataloader checkpoint not found at "
                f"{dataloader_path}. Sampler state will not be restored, which may lead to "
                f"data inconsistency."
            )
