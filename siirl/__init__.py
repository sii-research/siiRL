# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from importlib.metadata import version
from packaging.version import parse as parse_version
from importlib.metadata import PackageNotFoundError

from siirl.workers.databuffer import DataProto
from siirl.utils.extras.device import is_npu_available
from siirl.utils.logger.logging_utils import set_basic_config


set_basic_config()


__all__ = ["DataProto"]

if os.getenv("SIIRL_USE_MODELSCOPE", "False").lower() == "true":
    import importlib

    if importlib.util.find_spec("modelscope") is None:
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")
    # Patch hub to download models from modelscope to speed up.
    from modelscope.utils.hf_util import patch_hub

    patch_hub()

if is_npu_available:
    from .models.transformers import npu_patch as npu_patch

    package_name = "transformers"
    required_version_spec = "4.52.4"
    try:
        installed_version = version(package_name)
        installed = parse_version(installed_version)
        required = parse_version(required_version_spec)

        if not installed >= required:
            raise ValueError(f"{package_name} version >= {required_version_spec} is required on ASCEND NPU, current version is {installed}.")
    except PackageNotFoundError:
        raise ImportError(f"package {package_name} is not installed, please run pip install {package_name}=={required_version_spec}")
