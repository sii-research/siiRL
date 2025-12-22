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
from typing import Any
import transformers
from omegaconf import OmegaConf, DictConfig

from siirl.utils.params.training_args import SiiRLArguments


def _set_transformers_logging() -> None:
    if os.getenv("SIIRL_LOG_VERBOSITY", "INFO") in ["DEBUG", "INFO"]:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def parse_config(dict_config: DictConfig) -> SiiRLArguments:
    """Parse configuration using OmegaConf and convert to a SiiRLArguments instance."""
    # Convert OmegaConf config to a dictionary
    siirl_config_dict = OmegaConf.to_container(dict_config, resolve=True)

    # Recursively convert nested configs
    def convert_to_dataclass(obj: Any, dataclass_type: Any):
        if isinstance(obj, dict):
            fields = dataclass_type.__dataclass_fields__
            kwargs = {}
            for name, field_type in fields.items():
                if name in obj:
                    # Handle nested dataclasses
                    if hasattr(field_type.type, "__dataclass_fields__"):
                        kwargs[name] = convert_to_dataclass(obj[name], field_type.type)
                    else:
                        kwargs[name] = obj[name]
            return dataclass_type(**kwargs)
        return obj

    # Convert root config
    siirl_args = convert_to_dataclass(siirl_config_dict, SiiRLArguments)
    _set_transformers_logging()
    return siirl_args
