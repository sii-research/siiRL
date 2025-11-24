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

import gymnasium as gym
import numpy as np
import torch
from typing import Any, Dict, Tuple, Optional, List, Union
import warnings

try:
    import mani_skill.envs
except ImportError:
    pass

from siirl.workers.environment.embodied.base import BaseVLAEnvironment


class ManiSkillAdapter(BaseVLAEnvironment):
    """
    Adapter for ManiSkill environments with vectorized support.

    Behavior summary:
      - If num_envs == 1, create a single gym environment via gym.make().
      - If num_envs > 1, create a vectorized environment using SyncVectorEnv.
      - Observations are returned in a VLA-friendly format:
          * pixel_values: torch.FloatTensor normalized to [0,1], NCHW
          * depth (if available): torch.FloatTensor [1,H,W] or [N,1,H,W]
          * text: instruction (string) or list of strings for vectorized case (length N)
          * proprio: numpy array or None; for vectorized, stacked along axis 0 when possible.
      - reset() returns (processed_obs, infos) following Gymnasium API.
      - step(action) accepts batched or single actions (torch or numpy).
    """

    def __init__(self, task_name: str, num_envs: int = 1, **kwargs):
        """
        Args:
          task_name: name of ManiSkill task (e.g., "PickCube-v1").
          num_envs: number of parallel envs.
          **kwargs: passed to gym.make.
        """
        self.task_name = task_name
        self.num_envs = int(num_envs)
        self._env_kwargs = dict(
            obs_mode="rgbd",
            control_mode="pd_joint_delta_pos",
            render_mode="rgb_array",
            **kwargs,
        )

        self.env = gym.make(self.task_name, num_envs=self.num_envs, **self._env_kwargs)
        
        self.is_vector_env = self.num_envs > 1
        
        try:
            self.device = self.env.get_wrapper_attr("device")
        except AttributeError:
            self.device = getattr(self.env.unwrapped, "device", torch.device("cpu"))

        # Text instruction mapping
        self.instruction = self._get_task_instruction(task_name)

    def _get_task_instruction(self, task_name: str) -> str:
        """Return a short English instruction for the given task name."""
        mapping = {
            "PickCube-v1": "Pick up the red cube and move it to the green goal.",
            "StackCube-v1": "Stack the red cube on top of the green cube.",
        }
        return mapping.get(task_name, "Complete the task.")

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Any]:
        """
        Returns:
            processed_obs: Dict with tensors on GPU (pixel_values: [N, 3, H, W]).
            infos: Dict or List[Dict]
        """
        obs, infos = self.env.reset(**kwargs)
        processed = self._process_obs(obs)
        return processed, infos

    def step(self, action: Any) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Step with high-performance I/O.
        
        Args:
            action: torch.Tensor [N, ActionDim] on GPU, or numpy array (will be converted).
            
        Returns:
            All returns are torch.Tensors on the device (except infos).
        """
        # Ensure action is a Tensor on the correct device
        action_tensor = self._postprocess_action(action)
        
        # ManiSkill step returns: obs, reward, terminated, truncated, info
        # All are Tensors except info.
        obs, reward, terminated, truncated, infos = self.env.step(action_tensor)
        
        processed_obs = self._process_obs(obs)
        
        return processed_obs, reward, terminated, truncated, infos

    def _postprocess_action(self, action: Any) -> torch.Tensor:
        """Ensures action is a (N, Dim) tensor on the correct device."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        elif isinstance(action, list):
            action = torch.tensor(action)
            
        # Move to device if needed
        if action.device != self.device:
            action = action.to(self.device)

        # Ensure float32
        if action.dtype != torch.float32:
            action = action.float()

        # Handle Broadcasting if necessary (e.g., single action for all envs)
        # ManiSkill expects [N, Action_Dim]
        if action.ndim == 1 and self.num_envs > 1:
            # Broadcast: [Action_Dim] -> [N, Action_Dim]
            action = action.unsqueeze(0).expand(self.num_envs, -1)
            
        return action

    def _process_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process observations keeping everything on GPU.
        Expects `obs` to be a dict of Tensors (standard ManiSkill output).
        """
        rgb_tensor = None
        depth_tensor = None
        proprio_tensor = None

        # ManiSkill 3 structure: obs['sensor_data']['<camera_name>']['rgb'] or obs['image']...
        # We search for the first available RGB sensor.
        
        # Helper to find sensor data in nested dicts
        candidates = [obs]
        if "sensor_data" in obs: candidates.append(obs["sensor_data"])
        if "image" in obs: candidates.append(obs["image"])
        
        # Preferred camera order
        cam_names = ["base_camera", "hand_camera", "agent_camera", "camera"]
        
        found_cam = False
        for source in candidates:
            if found_cam: break
            # Try preferred names first
            for name in cam_names:
                if name in source and isinstance(source[name], dict) and "rgb" in source[name]:
                    rgb_tensor = source[name]["rgb"] # Shape: [N, H, W, C]
                    depth_tensor = source[name].get("depth")
                    found_cam = True
                    break
            
            # If not found, iterate all keys
            if not found_cam:
                for k, v in source.items():
                    if isinstance(v, dict) and "rgb" in v:
                        rgb_tensor = v["rgb"]
                        depth_tensor = v.get("depth")
                        found_cam = True
                        break
        
        # Fallback for flat structure
        if rgb_tensor is None:
             if "rgb" in obs: rgb_tensor = obs["rgb"]
             if "pixel_values" in obs: rgb_tensor = obs["pixel_values"]

        if rgb_tensor is None:
            # Extreme fallback: render() (Warning: this might be slow if not utilizing GPU render correctly)
            # In native vectorization, env.render() usually returns a CPU numpy array or list.
            try:
                rgb_cpu = self.env.render()
                if rgb_cpu is not None:
                    rgb_tensor = torch.tensor(rgb_cpu, device=self.device)
                else:
                    raise ValueError("Render returned None")
            except Exception:
                raise RuntimeError("Could not find 'rgb' in observation and render() failed.")

        # Expected incoming shape: [N, H, W, C] (ManiSkill standard)
        # Target shape: [N, C, H, W]
        
        # Sanity check for dimensions
        if rgb_tensor.ndim == 3: 
            # Case: [H, W, C] (Single env w/o batch dim, rare in native vec but possible)
            rgb_tensor = rgb_tensor.unsqueeze(0) # -> [1, H, W, C]
            
        # Permute: [N, H, W, C] -> [N, C, H, W]
        if rgb_tensor.shape[-1] == 3:
            rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
            
        # Normalize: [0, 255] (uint8) -> [0, 1] (float)
        if rgb_tensor.dtype == torch.uint8:
            rgb_tensor = rgb_tensor.float() / 255.0
        elif rgb_tensor.max() > 1.0 + 1e-6: # Add small epsilon for float comparison
            rgb_tensor = rgb_tensor.float() / 255.0
            
        processed_rgb = rgb_tensor.contiguous() # Ensure memory layout is efficient

        processed_depth = None
        if depth_tensor is not None:
            # Expected: [N, H, W, 1] or [N, H, W]
            if depth_tensor.ndim == 3:
                depth_tensor = depth_tensor.unsqueeze(-1) # -> [N, H, W, 1]
            
            # Permute -> [N, 1, H, W]
            processed_depth = depth_tensor.permute(0, 3, 1, 2).float().contiguous()

        # ManiSkill: obs['agent']['qpos']
        agent_data = obs.get("agent")
        if agent_data is not None:
            proprio_tensor = agent_data.get("qpos")
        
        if proprio_tensor is None:
             # Try other common keys
             for k in ["qpos", "joint_positions", "joint_pos"]:
                 if k in obs:
                     proprio_tensor = obs[k]
                     break
        
        if proprio_tensor is not None:
             if proprio_tensor.ndim == 1:
                 proprio_tensor = proprio_tensor.unsqueeze(0)
             proprio_tensor = proprio_tensor.float()

        # Text is still list of strings (on CPU), as tokenizers usually run on CPU 
        # or handle lists specifically.
        texts = [self.instruction] * self.num_envs

        result = {
            "pixel_values": processed_rgb, # torch.Tensor(GPU)
            "text": texts,                 # List[str]
            "proprio": proprio_tensor,     # torch.Tensor(GPU)
        }
        
        if processed_depth is not None:
            result["depth"] = processed_depth

        return result

    def close(self):
        """Close underlying env(s)."""
        try:
            if hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass
