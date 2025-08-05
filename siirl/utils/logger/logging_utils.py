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

from loguru import logger
import os
import sys

SIIRL_LOG_DIRCTORY = os.getenv("SIIRL_LOG_DIRECTORY", "siirl_logs")

SIIRL_LOGGING_FILENAME = os.getenv("SIIRL_LOGGING_FILENAME", "siirl")
if SIIRL_LOGGING_FILENAME != "":
    SIIRL_LOGGING_FILENAME += "_"


def set_basic_config():
    """
    This function sets the global logging format and level. It will be called when import siirl
    """
    logger.remove()
    logger.level("CRITICAL", color="<bold white on red>")
    # logger.level("ERROR", color="<red><bold>")
    # logger.level("WARNING", color="<yellow>")
    
    # Get log level from environment variable, default to INFO
    console_log_level = os.getenv("SIIRL_LOG_VERBOSITY", "INFO").upper()
    
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level.icon}</level> "
            "<level>{level: <8}</level> | "
            # Added :<magenta>{line}</magenta> to include the line number
            "<blue>{file}</blue>:<magenta>{line}</magenta>:<cyan>{function}</cyan> >> "
            "<level>{message}</level>"
        ),
        enqueue=True,
        colorize=True,
        level=console_log_level,
    )
    os.makedirs(SIIRL_LOG_DIRCTORY, exist_ok=True)

    logger.add(
        sink=os.path.join(SIIRL_LOG_DIRCTORY, SIIRL_LOGGING_FILENAME + "{time:YYYY-MM-DD-HH}.log"),
        rotation="500 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {module}:{function}:{line} >> {message}",
        compression="zip",
        encoding="utf-8",
        level="DEBUG",
        enqueue=True,
    )
