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

from mbridge.core.util import unwrap_model
from mbridge.core import Bridge
import torch


def load_weights_patch(
    self,
    models: list[torch.nn.Module],
    weights_path: str,
    memory_efficient: bool = False,
) -> None:
    """
    Load weights from a Hugging Face model into a Megatron-Core model.

    Args:
        models: List of model instances, supporting VPP (Virtual Pipeline Parallelism)
        weights_path: Path to the weights file or Hugging Face model identifier
    """
    self.safetensor_io = self._get_safetensor_io(weights_path)

    for i, model in enumerate(models):
        # map local weight names to global weight names
        local_to_global_map = self._weight_name_mapping_mcore_local_to_global(model)
        # map local weight names to huggingface weight names
        local_to_hf_map = {
            k: self._weight_name_mapping_mcore_to_hf(v)
            for k, v in local_to_global_map.items()
            if "_extra_state" not in k
        }
        # only tp_rank0/etp_rank0 load from disk, others load from tp_rank0/etp_rank0
        to_load_from_disk = []
        for local_name, hf_names in local_to_hf_map.items():
            if ".mlp.experts.linear_fc" in local_name:
                if self.mpu.etp_rank == 0:
                    to_load_from_disk.extend(hf_names)
            else:
                if self.mpu.tp_rank == 0:
                    to_load_from_disk.extend(hf_names)
                else:
                    # special case for lm_head.weight
                    # if make value model, every tp rank will load lm_head.weight
                    if "lm_head.weight" in hf_names:
                        to_load_from_disk.extend(hf_names)

        # load huggingface weights
        if not memory_efficient:
            hf_weights_map = self.safetensor_io.load_some_hf_weight(
                to_load_from_disk
            )
        model = unwrap_model(model)
        # Some weights are in named_parameters but not in state_dict.
        with torch.no_grad():
            for local_name, hf_names in local_to_hf_map.items():
                # Maybe a bug in torch_npu. Some weights are registered in named_parameters but not in state_dict.
                if model.state_dict().get(local_name, None) is None:
                    param = dict(model.named_parameters())[local_name]
                else: 
                    param = model.state_dict()[local_name]
                # hf format to mcore format
                if set(to_load_from_disk) & set(hf_names):
                    if not memory_efficient:
                        hf_weights = [hf_weights_map[x] for x in hf_names]
                    else:
                        hf_weights = [
                            self.safetensor_io.load_one_hf_weight(x) for x in hf_names
                        ]
                    mcore_weight = self._weight_to_mcore_format(local_name, hf_weights)
                else:
                    mcore_weight = None
                if hf_names[0] == "lm_head.weight":
                    if param.shape[0] == 1 and mcore_weight.shape[0] != 1:
                        # skip lm_head.weight when the model is a value model
                        continue

                param_to_load = torch.empty_like(param)
                if ".mlp.experts.linear_fc" in local_name:
                    # split mcore weights across etp
                    if self.mpu.etp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.etp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous()
                            for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.etp_group, 0),
                        group=self.mpu.etp_group,
                    )
                else:
                    # split mcore weights across tp
                    if self.mpu.tp_rank == 0:
                        mcore_weights_tp_split = self._weight_split_across_tp(
                            local_name, mcore_weight, param, self.mpu.tp_size
                        )
                        mcore_weights_tp_split = list(mcore_weights_tp_split)
                        mcore_weights_tp_split = [
                            t.to(param.device, dtype=param.dtype).contiguous()
                            for t in mcore_weights_tp_split
                        ]
                    else:
                        mcore_weights_tp_split = None
                    torch.distributed.scatter(
                        param_to_load,
                        mcore_weights_tp_split,
                        src=torch.distributed.get_global_rank(self.mpu.tp_group, 0),
                        group=self.mpu.tp_group,
                    )
                # load
                param.copy_(param_to_load.detach())

def _weight_name_mapping_mcore_local_to_global_patch(
    self, model: torch.nn.Module, consider_ep: bool = True
) -> dict[str, str]:
    """
    Map local weight names to global weight names, supporting VPP and EP.

    Args:
        model: The model instance

    Returns:
        dict: Mapping from local weight names to global weight names
    """
    # vpp
    local_layer_to_global_layer = {}
    model = unwrap_model(model)
    if hasattr(model, "decoder"):
        for idx, layer in enumerate(model.decoder.layers):
            local_layer_to_global_layer[idx] = layer.layer_number - 1
    # Maybe a bug in torch_npu. Some weights are registered in named_parameters but not in state_dict.
    all_named_param_names = [
        k for k,_ in model.named_parameters() if "_extra_state" not in k
    ]
    all_state_dict_keys = [
        k for k in model.state_dict().keys() if "_extra_state" in k
    ]
    all_param_names = list(dict.fromkeys(all_named_param_names + all_state_dict_keys))
    ret = {}
    for param_name in all_param_names:
        keyword = "decoder.layers."
        if keyword in param_name:
            layer_idx = int(param_name.split(keyword)[1].split(".")[0])
            global_layer_idx = local_layer_to_global_layer[layer_idx]
            ret[param_name] = param_name.replace(
                f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
            )
        else:
            ret[param_name] = param_name

    # ep
    if self.mpu.ep_size > 1 and consider_ep:
        num_experts = self.config.num_moe_experts
        num_experts_per_rank = num_experts // self.mpu.ep_size
        local_expert_to_global_expert = {
            i: i + num_experts_per_rank * self.mpu.ep_rank
            for i in range(num_experts_per_rank)
        }
        for k in ret.keys():
            v = ret[k]
            if ".mlp.experts.linear_fc" in v:
                name_prefix, local_expert_id = v.split(".weight")
                global_expert_idx = local_expert_to_global_expert[
                    int(local_expert_id)
                ]
                ret[k] = f"{name_prefix}.weight{global_expert_idx}"

    return ret

Bridge.load_weights = load_weights_patch
Bridge._weight_name_mapping_mcore_local_to_global = _weight_name_mapping_mcore_local_to_global_patch
