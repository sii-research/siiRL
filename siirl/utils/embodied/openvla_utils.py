"""Utils for evaluating OpenVLA or fine-tuned OpenVLA policies."""

import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import tensorflow as tf
import torch
from loguru import logger
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Apply JSON numpy patch for serialization
json_numpy.patch()

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Uses file locking and atomic write to ensure thread-safety and prevent corruption.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        logger.warning(f"No config.json found at {config_path}")
        return

    import fcntl
    import tempfile
    
    lock_path = os.path.join(pretrained_checkpoint, ".config.json.lock")
    max_retries = 5
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # Acquire file lock for safe concurrent access
            with open(lock_path, 'w') as lock_file:
                # Wait up to 30 seconds for the lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                
                try:
                    # Re-check if config already has correct auto_map (another process may have updated it)
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    
                    expected_auto_map = {
                        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
                        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
                    }
                    
                    # If already correctly configured, skip update
                    if config.get("auto_map") == expected_auto_map:
                        logger.info(f"config.json already has correct auto_map, skipping update")
                        return
                    
                    # Create timestamped backup (only if we need to update)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
                    shutil.copy2(config_path, backup_path)
                    logger.info(f"Created backup of original config at: {os.path.abspath(backup_path)}")

                    # Update the config
                    config["auto_map"] = expected_auto_map

                    # Atomic write: write to temp file, then rename
                    # This ensures readers never see a partial/corrupted file
                    fd, temp_path = tempfile.mkstemp(
                        dir=pretrained_checkpoint, 
                        prefix=".config.json.tmp.",
                        suffix=".json"
                    )
                    try:
                        with os.fdopen(fd, 'w') as f:
                            json.dump(config, f, indent=2)
                            f.flush()
                            os.fsync(f.fileno())  # Ensure data is written to disk
                        
                        # Atomic rename - this is atomic on POSIX systems
                        os.replace(temp_path, config_path)
                        
                        logger.info(f"Updated config.json at: {os.path.abspath(config_path)}")
                        logger.info("Changes made:")
                        logger.info('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
                        logger.info('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')
                        return
                    except Exception as e:
                        # Clean up temp file if something went wrong
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                        raise
                finally:
                    # Release lock
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    
        except (json.JSONDecodeError, IOError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(f"Failed to update config.json after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error updating config.json: {e}")
            raise


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            logger.info(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            logger.info(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            logger.info(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            logger.info(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        logger.info(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            logger.warning(f"`{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)







