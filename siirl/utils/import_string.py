# Copyright 2025, Infrawaves. All rights reserved.
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


def import_string(import_name: str):
    """Dynamically imports a module or object from a string."""
    module_name, obj_name = import_name.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {import_name}") from e

if __name__ == "__main__":
    print(import_string("siirl.engine.sharding_manager.fsdp_vllm.MultiAgentFSDPVLLMShardingManager"))