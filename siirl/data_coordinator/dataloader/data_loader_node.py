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

from typing import Any, Dict, Iterator, Optional

import torch
from loguru import logger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from siirl.execution.dag import Node, NodeRole, NodeStatus, NodeType
from siirl.models.loader import load_tokenizer
from siirl.global_config.params import SiiRLArguments

from siirl.data_coordinator.dataloader.partitioned_dataset import PartitionedRLHFDataset


class RepeatDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that repeats the base dataset multiple times.

    This class allows you to virtually extend the length of a given dataset by repeating its samples
    a specified number of times. It is useful for scenarios where you want to train for more epochs
    without reloading or reshuffling the data, or to balance datasets by oversampling.

    Args:
        base_dataset (torch.utils.data.Dataset): The original dataset to be repeated.
        repeat_factor (int): The number of times to repeat the base dataset.

    Attributes:
        base_dataset (torch.utils.data.Dataset): The original dataset.
        repeat_factor (int): The number of repetitions.
        length (int): The total length of the repeated dataset.

    Example:
        >>> base_dataset = MyCustomDataset()
        >>> repeated_dataset = RepeatDataset(base_dataset, repeat_factor=3)
        >>> len(repeated_dataset) == 3 * len(base_dataset)
        True

    """

    def __init__(self, base_dataset, repeat_factor):
        self.base_dataset = base_dataset
        self.repeat_factor = repeat_factor
        self.length = len(base_dataset) * repeat_factor

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.base_dataset[idx % len(self.base_dataset)]


class DataLoaderNode(Node):
    """
    Represents a data loader node in the DAG.
    This version uses the PartitionedRLHFDataset for efficient, memory-safe
    distributed data loading. Each rank only loads and processes its own data slice.
    """

    def __init__(self, node_id: str, global_config: SiiRLArguments, config: Optional[Dict[str, Any]] = None, retry_limit: int = 0):
        """
        Initialize a data loader node.

        Args:
            node_id (str): The unique identifier of the node.
            global_config(SiiRLArguments): The arguments from config file.
            config (Optional[Dict[str, Any]]): Specific configuration information for the node. Defaults to an empty dictionary.
            retry_limit (int): The maximum number of retries when the node execution fails. Defaults to 0 (no retries).
        """
        super().__init__(node_id, NodeType.DATA_LOAD, NodeRole.DEFAULT, config=config, retry_limit=retry_limit)
        self.global_config = global_config

        if "tokenizer" in self.config:
            self.tokenizer = self.config["tokenizer"]
            self.processor = self.config["processor"]
        else:
            # Load tokenizer and processor
            tokenizer_module = load_tokenizer(model_args=global_config.actor_rollout_ref.model)
            self.tokenizer = tokenizer_module["tokenizer"]
            self.processor = tokenizer_module["processor"]
            
        # force load in main process for vision language model
        self.num_loader_workers = 0 if global_config.actor_rollout_ref.rollout.name == "sglang" or self.processor is not None \
            else config.get("num_loader_workers", 8)
        
        # Get group world size, rank, parallel size from config.
        #   Group world size means the rollout pytorch distributed group total gpus.
        #   Group rank means the process index in distributed group.
        #   Group parallel size means the rollout total parallel size, e.g. tp_size * pp_size
        self.group_world_size = config["group_world_size"]
        self.group_rank = config["group_rank"]
        self.group_parallel_size = config["group_parallel_size"]
        if self.group_world_size % self.group_parallel_size != 0:
            # Log an error or raise an exception if world_size is not divisible by group_parallel_size
            error_msg = f"group_world_size ({self.group_world_size}) must be divisible by group_parallel_size ({self.group_parallel_size})."
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Calculate the world size and rank for rollout data parallelism, which is actually needed for data partitioning.
        self.rollout_ddp_world_size = self.group_world_size // self.group_parallel_size
        self.rollout_ddp_rank = self.group_rank // self.group_parallel_size

        self._current_train_iter: Optional[Iterator] = None
        self._current_val_iter: Optional[Iterator] = None
        self._current_epoch: int = -1

        self._create_dataloader()

        self.num_train_batches = len(self.train_dataloader) if self.train_dataloader else 0
        self.num_val_batches = len(self.val_dataloader) if self.val_dataloader else 0

        logger.info(f"DataLoaderNode '{self.node_id}' initialized:")
        logger.info(f"  Group rank: {self.group_rank} / {self.group_world_size}")
        logger.info(f"  Rollout DDP rank: {self.rollout_ddp_rank} / {self.rollout_ddp_world_size}")
        logger.info(f"  Train batches per epoch for this rank: {self.num_train_batches}")
        logger.info(f"  Total training steps (approx): {self.total_training_steps}")

    def _create_dataloader(self):
        """
        Initializes and configures the training and validation DataLoaders for RLHF tasks.

        When enable `auto_repeat`, if the dataset is too small to form a batch, it will be automatically repeated
        until at least one batch can be formed.

        This method performs the following steps:
        1. Creates the training dataset (`PartitionedRLHFDataset`) with the provided configuration, tokenizer, processor, and distributed data parallel (DDP) settings.
        2. Sets up the sampler for the training DataLoader:
            - Uses a `RandomSampler` with a seeded generator if shuffling is enabled in the configuration.
            - Uses a `SequentialSampler` otherwise.
        3. Configures the tokenizer's padding side to "left" to ensure correct sequence alignment.
        4. Creates the training DataLoader (`StatefulDataLoader`) with the specified batch size, number of workers, sampler, and collator.
        5. Creates the validation dataset and DataLoader, using the full dataset as a single batch for evaluation.
        6. Asserts that the training DataLoader contains at least one batch.
        7. Calculates the total number of training steps based on the number of batches and epochs, or uses a user-specified value if provided.
        8. Updates the total training steps in the optimizer configurations for both the actor and critic components.
        """
        # Create the partitioned training dataset for this rank
        self.train_dataset = PartitionedRLHFDataset(config=self.global_config, tokenizer=self.tokenizer, processor=self.processor, ddp_rank=self.rollout_ddp_rank, ddp_world_size=self.rollout_ddp_world_size, is_eval=False, drop_last=self.config.get("train_drop_last", True))

        # Calculate batch size per rank
        train_batch_size = self.global_config.data.train_batch_size // self.rollout_ddp_world_size

        # Auto-repeat logic: if dataset is too small, repeat it until at least one batch can be formed
        auto_repeat = self.config.get("auto_repeat", False)
        train_len = len(self.train_dataset)
        if auto_repeat and train_len < train_batch_size:
            repeat_factor = (train_batch_size + train_len - 1) // train_len

            self.train_dataset = RepeatDataset(self.train_dataset, repeat_factor)
            logger.warning(f"Rank {self.rollout_ddp_rank}: Training dataset too small (size={train_len}), auto-repeating {repeat_factor} times to ensure at least one batch (batch_size={train_batch_size}). Now RepeatDataset size={len(self.train_dataset)}")

        # Choose sampler: RandomSampler with seed if shuffle enabled, else SequentialSampler
        if self.global_config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.global_config.trainer.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        # Create the training dataloader with the specified batch size, workers, sampler, and collator
        from siirl.data_coordinator.dataloader.partitioned_dataset import collate_fn as default_collate_fn

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset, batch_size=train_batch_size, num_workers=self.num_loader_workers, drop_last=True, collate_fn=default_collate_fn, sampler=sampler)

        # Create the partitioned validation dataset for this rank
        self.val_dataset = PartitionedRLHFDataset(config=self.global_config, tokenizer=self.tokenizer, processor=self.processor, ddp_rank=self.rollout_ddp_rank, ddp_world_size=self.rollout_ddp_world_size, is_eval=True, drop_last=self.config.get("eval_drop_last", False))

        # Create the validation dataloader, loading the entire validation set as one batch
        val_batch_size = self.global_config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(dataset=self.val_dataset, batch_size=val_batch_size, num_workers=self.num_loader_workers, shuffle=False, drop_last=False, collate_fn=default_collate_fn)

        # Assert that there is at least one batch for this rank
        assert len(self.train_dataloader) >= 1, f"Not enough data for current rank (rank id: {self.rollout_ddp_rank}) to consume. Please increase the train datasets or reduce the number of GPUs."
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        # Calculate the number of batches and total training steps
        num_batches = len(self.train_dataloader) if self.train_dataloader else 0
        total_training_steps = num_batches * self.global_config.trainer.total_epochs
        # Use user-specified total_training_steps if provided
        if self.global_config.trainer.total_training_steps is not None:
            total_training_steps = self.global_config.trainer.total_training_steps

        self.total_training_steps = total_training_steps

        # Update total training steps in optimizer configs for actor and critic
        self.global_config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        self.global_config.critic.optim.total_training_steps = total_training_steps
        
        # Indicates the samples for this rank has already been expand
        self.is_val_trailing_rank = self.val_dataset.is_trailing_rank

    def get_train_dataloader(self):
        """
        Returns the training data loader.

        Returns:
            DataLoader: The data loader used for training.
        """
        return self.train_dataloader

    def get_val_dataloader(self):
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: The dataloader used for validation data.
        """
        return self.val_dataloader

    def run(self, epoch: Optional[int] = None, is_validation_step: bool = False, **kwargs: Any) -> Any:
        """
        Executes the data loading process for a given step or validation.

        Args:
            epoch (Optional[int]): The current epoch number. Required for training steps to handle
                                   sampler state (e.g., DistributedSampler.set_epoch()).
            is_validation_step (bool): Flag indicating if validation data is requested.
            **kwargs: Additional arguments (not used directly in this basic version but
                      part of the Node.execute signature).

        Returns:
            Any: A batch of data. The structure depends on the collate_fn.

        Raises:
            ValueError: If epoch is not provided for a training step.
            StopIteration: If the dataloader is exhausted and cannot provide more data
                           (though this might be handled by the DAG scheduler).
        """
        self.update_status(NodeStatus.RUNNING)
        logger.debug(f"Node {self.node_id} execute: epoch={epoch}, is_validation_step={is_validation_step}")

        try:
            if is_validation_step:
                if not self.val_dataloader:  # Handles empty validation dataset
                    logger.warning(f"Rank {self.group_rank}: Validation dataloader is not available or empty.")
                    self.update_status(NodeStatus.COMPLETED)  # Or FAILED if this is an error condition
                    return None  # Or an empty batch marker

                # Validation dataloader loads the entire validation set as one batch.
                # We get a fresh iterator each time for validation.
                if self._current_val_iter is None:
                    self._current_val_iter = iter(self.val_dataloader)

                try:
                    batch = next(self._current_val_iter)
                    logger.debug(f"Node {self.node_id}: Yielding validation batch.")
                    # Reset for next validation call, as it's one batch
                    self._current_val_iter = None
                except StopIteration:
                    logger.warning(f"Node {self.node_id}: Validation dataloader exhausted unexpectedly (should be one batch). Resetting.")
                    # This case should ideally not happen if batch_size = len(dataset) and it's not empty
                    self._current_val_iter = iter(self.val_dataloader)  # Get a fresh iterator
                    try:
                        batch = next(self._current_val_iter)
                    except StopIteration:
                        logger.error(f"Node {self.node_id}: Validation dataloader is empty even after reset.")
                        self.update_status(NodeStatus.FAILED, "Validation dataloader empty")
                        return None
            else:  # Training step
                if epoch is None:
                    error_msg = "Epoch must be provided for training steps."
                    logger.error(f"Node {self.node_id}: {error_msg}")
                    self.update_status(NodeStatus.FAILED, error_msg)
                    raise ValueError(error_msg)

                if not self.train_dataloader:  # Handles empty training dataset
                    logger.warning(f"Rank {self.group_rank}: Training dataloader is not available or empty.")
                    self.update_status(NodeStatus.COMPLETED)  # Or FAILED
                    return None  # Or an empty batch marker

                if self._current_epoch != epoch or self._current_train_iter is None:
                    logger.info(f"Node {self.node_id}: New epoch ({epoch}) or first step. Initializing train iterator.")
                    self._current_epoch = epoch
                    # Set epoch for DistributedSampler if applicable
                    if hasattr(self.train_dataloader.sampler, "set_epoch") and isinstance(self.train_dataloader.sampler, DistributedSampler):
                        logger.debug(f"Node {self.node_id}: Setting epoch {epoch} for DistributedSampler.")
                        self.train_dataloader.sampler.set_epoch(epoch)

                    self._current_train_iter = iter(self.train_dataloader)

                try:
                    batch = next(self._current_train_iter)
                    logger.debug(f"Node {self.node_id}: Yielding training batch for epoch {epoch}.")
                except StopIteration:
                    # This means the current epoch's data is exhausted.
                    # The DAG scheduler should ideally handle this by moving to the next epoch
                    # or terminating if all epochs are done.
                    # For this node, it signals completion for this particular call if data is expected.
                    error_msg = f"Training dataloader exhausted for epoch {epoch}. This might be expected at epoch end."
                    logger.info(f"Node {self.node_id}: {error_msg}")
                    # We might not want to mark FAILED here, as it's a natural end of an iterator.
                    # The caller (DAG executor) should decide if more data was expected.
                    # For now, let's re-raise StopIteration to signal the caller.
                    self.update_status(NodeStatus.COMPLETED)  # Or a custom status like 'EPOCH_END'
                    raise  # Re-raise StopIteration

            self.update_status(NodeStatus.COMPLETED)
            return batch

        except Exception as e:
            error_msg = f"Error during data loading in node {self.node_id}: {e}"
            logger.exception(error_msg)  # Log with stack trace
            self.update_status(NodeStatus.FAILED, str(e))
            raise  # Re-raise the exception so the DAG executor can handle it

    def state_dict(self) -> Dict[str, Any]:
        """
        Captures the state of the DataLoaderNode, primarily the training dataloader's state.

        Returns:
            Dict[str, Any]: A dictionary containing the node's state.
        """
        return {
            "train_dataloader_state": self.train_dataloader.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Restores the state of the DataLoaderNode from a state dictionary.

        Args:
            state_dict (Dict[str, Any]): The state dictionary to load.
        """
        if "train_dataloader_state" in state_dict:
            self.train_dataloader.load_state_dict(state_dict["train_dataloader_state"])
            # After loading state, the current iterator is invalid because it's tied to the old
            # sampler state. Setting it to None forces the run() method to create a new,
            # valid iterator that is synchronized with the restored state.
            self._current_train_iter = None
            logger.info(f"Node {self.node_id} (Rank {self.group_rank}): Successfully loaded train_dataloader state. Iterator will be reset on next call.")
