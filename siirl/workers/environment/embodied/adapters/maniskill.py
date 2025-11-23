
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
from PIL import Image
from typing import Any, Dict, Tuple, Optional, List, Callable

try:
    import mani_skill.envs
except ImportError:
    pass

from siirl.workers.environment.embodied.base import BaseVLAEnvironment


def _make_env_fn(task_name: str, kwargs: Dict[str, Any]) -> Callable[[], gym.Env]:
    """Return a callable that constructs a single environment instance (for vectorized envs)."""
    def _thunk():
        return gym.make(task_name, **kwargs)
    return _thunk


class ManiSkillAdapter(BaseVLAEnvironment):
    """
    Adapter for ManiSkill environments with vectorized support.

    Behavior summary:
      - If num_envs == 1, create a single gym environment via gym.make().
      - If num_envs > 1, create a vectorized environment using SyncVectorEnv.
      - Observations are returned in a VLA-friendly format:
          * pixel_values: torch.FloatTensor normalized to [0,1], CHW
            - single env: [3,H,W]
            - vectorized: [N,3,H,W]
          * depth (if available): torch.FloatTensor [1,H,W] or [N,1,H,W]
          * text: instruction (string) or list of strings for vectorized case (length N)
          * proprio: numpy array or None; for vectorized, stacked along axis 0 when possible.
      - reset() returns (processed_obs, infos) following Gymnasium API.
      - step(action) accepts batched or single actions (torch or numpy).
    """

    def __init__(self, task_name: str, max_episode_steps: int = 100, num_envs: int = 1, **kwargs):
        """
        Args:
          task_name: name of ManiSkill task, e.g., "PickCube-v1".
          max_episode_steps: maximum steps per episode (passed to superclass if used).
          num_envs: number of parallel envs. If 1, use gym.make; otherwise build SyncVectorEnv.
          **kwargs: forwarded to gym.make for each env (e.g., obs_mode, control_mode).
        """
        super().__init__(task_name, max_episode_steps, **kwargs)
        self.task_name = task_name
        self.num_envs = int(num_envs)
        self._env_kwargs = dict(obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="rgb_array", **kwargs)

        if self.num_envs <= 1:
            # single environment
            self.env = gym.make(self.task_name, **self._env_kwargs)
            self.is_vector_env = False
        else:
            # vectorized environments: create a list of callables for SyncVectorEnv
            env_fns = [_make_env_fn(self.task_name, self._env_kwargs) for _ in range(self.num_envs)]
            # You may switch to AsyncVectorEnv for true async behavior.
            self.env = gym.vector.SyncVectorEnv(env_fns)
            self.is_vector_env = True

        # Text instruction mapping (can be extended / replaced by dataset annotations)
        self.instruction = self._get_task_instruction(task_name)

    def _get_task_instruction(self, task_name: str) -> str:
        """Return a short English instruction for the given task name."""
        mapping = {
            "PickCube-v1": "Pick up the red cube and move it to the green goal.",
            "StackCube-v1": "Stack the red cube on top of the green cube.",
        }
        return mapping.get(task_name, "Complete the task.")

    def reset(self) -> Tuple[Dict[str, Any], Any]:
        """
        Reset env(s) and return (processed_obs, infos).
        - For single env: processed_obs fields have single examples (pixel_values: [3,H,W]).
        - For vectorized env: processed_obs fields are batched (pixel_values: [N,3,H,W]).
        - infos follows gymnasium semantics: single info dict or a list of infos for vectorized env.
        """
        # Gymnasium: env.reset() returns (obs, info) for both single and vectorized envs
        obs, infos = self.env.reset()
        processed = self._process_obs(obs, batch_mode=self.is_vector_env)
        return processed, infos

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Any]:
        """
        Step the environment(s) with action.
        Accepts a single action or batched actions:
          - If vectorized: action should be shape [N, ...] or broadcastable.
          - action may be torch.Tensor or numpy array.
        Returns:
          - processed_obs: batched or single observation dict
          - reward: scalar if single env, or numpy array of shape [N] for vectorized
          - terminated: bool or array-like
          - truncated: bool or array-like
          - infos: info dict or list of dicts
        """
        action_np = self._postprocess_action(action, batch_mode=self.is_vector_env)
        obs, reward, terminated, truncated, infos = self.env.step(action_np)
        processed = self._process_obs(obs, batch_mode=self.is_vector_env)
        # Maintain types: reward/terminated/truncated are returned as-is from the vector env
        return processed, reward, terminated, truncated, infos

    def _postprocess_action(self, action: Any, batch_mode: bool = False) -> np.ndarray:
        """
        Convert action (torch or numpy) to numpy array suitable for env.step().

        - If env.action_space is Box and actions appear normalized in [-1,1], automatically map to [low, high].
        - Works for single or batched actions. For vectorized envs, action shapes should be (N, action_dim) or broadcastable.
        """
        # Convert torch Tensor to numpy
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action)

        action_space = getattr(self.env, "action_space", None)
        # For vectorized envs, the action_space is the same for each env; it may still be a Box
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            low = action_space.low
            high = action_space.high
            eps = 1e-6

            # Ensure we handle batch dimension: if batch_mode, low/high shape correspond to single env (action_dim,)
            # Attempt to broadcast action to match expected shape
            try:
                # If vectorized and action lacks batch dim, broadcast it
                if batch_mode:
                    if action.ndim == 1:
                        # single action provided -> tile to all envs
                        action = np.broadcast_to(action, (self.num_envs,) + action.shape).copy()
                # Attempt to broadcast to low.shape (works when action shape matches)
                # If low has shape (action_dim,) and action has shape (N, action_dim), this is fine.
            except Exception:
                pass

            # Detect normalized [-1,1] actions: all values in [-1-eps, 1+eps]
            if np.all(action <= 1.0 + eps) and np.all(action >= -1.0 - eps):
                mid = (high + low) / 2.0
                half_range = (high - low) / 2.0
                try:
                    # If batched: action shape (N, action_dim), mid shape (action_dim,) -> broadcast OK
                    action = mid + action * half_range
                except Exception:
                    # Fall back to elementwise safe mapping
                    action = np.clip(action, -1.0, 1.0)

            # Clip to bounds
            action = np.clip(action, low, high)

        # Cast to float32 for most continuous control envs
        if np.issubdtype(action.dtype, np.floating):
            action = action.astype(np.float32)
        else:
            # convert ints to floats if necessary
            action = action.astype(np.float32)

        return action

    def _process_obs(self, obs: Any, batch_mode: bool = False) -> Dict[str, Any]:
        """
        Process raw observation(s) into the VLA-friendly format.
        Supports:
          - Single observation (np.ndarray, tensor, or dict)
          - Batched observation returned by vectorized envs (dict of arrays or array with leading dim)
        Returns:
          - For single env: pixel_values torch.Tensor [3,H,W]
          - For vectorized env: pixel_values torch.Tensor [N,3,H,W]
        """
        if batch_mode:
            # obs from vectorized env is typically a dict of arrays with leading batch dim
            return self._process_obs_batch(obs)
        else:
            # single observation
            return self._process_obs_single(obs)

    def _process_obs_single(self, obs: Any) -> Dict[str, Any]:
        """Process a single observation into pixel_values (torch CHW), depth (optional), text, proprio."""
        rgb = None
        depth = None
        proprio = None

        # Extract rgb/depth/proprio similar to previous robust logic
        if isinstance(obs, np.ndarray):
            rgb = obs
        elif isinstance(obs, torch.Tensor):
            rgb = obs.cpu().numpy()
        elif isinstance(obs, dict):
            # obs['image'] as dict of cameras
            if "image" in obs and isinstance(obs["image"], dict):
                cams = obs["image"]
                preferred = ["base_camera", "hand_camera", "agent_camera", "camera"]
                for name in preferred:
                    cam = cams.get(name)
                    if cam is None:
                        continue
                    if isinstance(cam, dict) and "rgb" in cam:
                        rgb = cam["rgb"]
                        depth = cam.get("depth", depth)
                        break
                    elif isinstance(cam, (np.ndarray, torch.Tensor)):
                        rgb = cam
                        break
                if rgb is None:
                    for cam_val in cams.values():
                        if isinstance(cam_val, dict) and "rgb" in cam_val:
                            rgb = cam_val["rgb"]
                            depth = cam_val.get("depth", depth)
                            break
                        if isinstance(cam_val, (np.ndarray, torch.Tensor)):
                            rgb = cam_val
                            break

            # top-level keys
            if rgb is None:
                for k in ("rgb", "image_rgb", "visual"):
                    if k in obs:
                        rgb = obs[k]
                        break

            if depth is None:
                for k in ("depth", "image_depth"):
                    if k in obs:
                        depth = obs[k]
                        break

            # proprio
            for root in ("agent", "robot", "state", "proprio", "robot_state"):
                if root in obs and isinstance(obs[root], dict):
                    qpos = (
                        obs[root].get("qpos")
                        or obs[root].get("joint_positions")
                        or obs[root].get("joint_pos")
                    )
                    if qpos is not None:
                        proprio = np.asarray(qpos, dtype=np.float32)
                        break
            if proprio is None:
                for k in ("qpos", "joint_positions", "joint_pos"):
                    if k in obs:
                        proprio = np.asarray(obs[k], dtype=np.float32)
                        break

        # fallback render
        if rgb is None:
            rgb = self.env.render()

        rgb = self._to_numpy(rgb)

        # CHW->HWC if necessary
        if rgb.ndim == 3 and rgb.shape[0] in (1, 3) and rgb.shape[0] != rgb.shape[2]:
            rgb = np.transpose(rgb, (1, 2, 0))

        # normalize to float32 [0,1]
        if rgb.dtype != np.float32:
            if rgb.max() <= 1.0:
                rgb = rgb.astype(np.float32)
            else:
                rgb = (rgb.astype(np.float32) / 255.0)

        # convert to CHW tensor
        rgb_chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

        depth_tensor = None
        if depth is not None:
            depth = self._to_numpy(depth).astype(np.float32)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = np.squeeze(depth, 0)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = np.squeeze(depth, 2)
            depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        processed = {
            "pixel_values": rgb_chw,
            "text": self.instruction,
            "proprio": proprio,
        }
        if depth_tensor is not None:
            processed["depth"] = depth_tensor

        return processed

    def _process_obs_batch(self, obs: Any) -> Dict[str, Any]:
        """
        Process batched observations returned by vectorized envs.

        Typical obs forms:
          - numpy array with leading dim N (image frames)
          - dict where each value is an array with leading dim N
        """
        # If obs is a dict: values are arrays with leading dim N
        if isinstance(obs, dict):
            # Try to build per-env single obs and call single-processor to reuse logic
            # But for efficiency, we'll try to vectorize image conversion.
            # Primary target: extract rgb batch and depth batch if available.
            rgb_batch = None
            depth_batch = None
            proprio_batch = None

            # 1) obs['image'] case
            if "image" in obs and isinstance(obs["image"], dict):
                cams = obs["image"]
                # Try preferred cameras in order; pick first camera that exists and has rgb
                for name in ("base_camera", "hand_camera", "agent_camera", "camera"):
                    cam = cams.get(name)
                    if cam is None:
                        continue
                    # cam could be dict with 'rgb' array
                    if isinstance(cam, dict) and "rgb" in cam:
                        rgb_batch = self._to_numpy(cam["rgb"])
                        if "depth" in cam:
                            depth_batch = self._to_numpy(cam["depth"])
                        break
                    elif isinstance(cam, (np.ndarray, torch.Tensor)):
                        rgb_batch = self._to_numpy(cam)
                        break
                # fallback: try any camera that contains 'rgb'
                if rgb_batch is None:
                    for cam_val in cams.values():
                        if isinstance(cam_val, dict) and "rgb" in cam_val:
                            rgb_batch = self._to_numpy(cam_val["rgb"])
                            depth_batch = self._to_numpy(cam_val.get("depth")) if cam_val.get("depth") is not None else None
                            break
                        if isinstance(cam_val, (np.ndarray, torch.Tensor)):
                            rgb_batch = self._to_numpy(cam_val)
                            break

            # 2) top-level rgb/depth keys
            if rgb_batch is None:
                for k in ("rgb", "image_rgb", "visual"):
                    if k in obs:
                        rgb_batch = self._to_numpy(obs[k])
                        break

            if depth_batch is None:
                for k in ("depth", "image_depth"):
                    if k in obs:
                        depth_batch = self._to_numpy(obs[k])
                        break

            # 3) proprio stacking
            # attempt to gather per-env proprio vectors into an array [N, ...]
            proprio_candidates = []
            for root in ("agent", "robot", "state", "proprio", "robot_state"):
                if root in obs and isinstance(obs[root], dict):
                    # obs[root] is dict-of-arrays (batched)
                    qpos_arr = obs[root].get("qpos") or obs[root].get("joint_positions") or obs[root].get("joint_pos")
                    if qpos_arr is not None:
                        try:
                            proprio_batch = np.asarray(qpos_arr, dtype=np.float32)
                            break
                        except Exception:
                            continue
            if proprio_batch is None:
                for k in ("qpos", "joint_positions", "joint_pos"):
                    if k in obs:
                        try:
                            proprio_batch = np.asarray(obs[k], dtype=np.float32)
                            break
                        except Exception:
                            pass

            # If rgb_batch is still None, fallback: env.render() per env is expensive; try single render (may return image for single env)
            if rgb_batch is None:
                # Try env.render() which for vector env may return a list/array
                try:
                    rgb_batch = self.env.render()
                except Exception:
                    raise RuntimeError("Unable to obtain batched RGB frames from vectorized observation or env.render().")

            # rgb_batch should now be ndarray with leading dim N
            rgb_batch = self._to_numpy(rgb_batch)
            if rgb_batch.ndim == 4 and rgb_batch.shape[1] in (1, 3) and rgb_batch.shape[1] != rgb_batch.shape[3]:
                # CHWWB? some shapes may be (N, C, H, W) -> convert to (N, H, W, C)
                rgb_batch = np.transpose(rgb_batch, (0, 2, 3, 1))

            # Normalize rgb batch to float32 [0,1]
            if rgb_batch.dtype != np.float32:
                if rgb_batch.max() <= 1.0:
                    rgb_batch = rgb_batch.astype(np.float32)
                else:
                    rgb_batch = (rgb_batch.astype(np.float32) / 255.0)

            # Convert to torch tensor [N, C, H, W]
            # Ensure HWC -> CHW per sample
            if rgb_batch.ndim == 4:
                # assume (N, H, W, C)
                rgb_t = torch.from_numpy(rgb_batch).permute(0, 3, 1, 2).contiguous()
            elif rgb_batch.ndim == 3:
                # single image repeated? add batch dim
                rgb_t = torch.from_numpy(rgb_batch).permute(2, 0, 1).unsqueeze(0).contiguous()
            else:
                raise RuntimeError(f"Unexpected rgb_batch shape: {rgb_batch.shape}")

            depth_t = None
            if depth_batch is not None:
                depth_batch = self._to_numpy(depth_batch).astype(np.float32)
                # handle shapes like (N,H,W,1) or (N,1,H,W)
                if depth_batch.ndim == 4 and depth_batch.shape[-1] == 1:
                    depth_batch = np.squeeze(depth_batch, axis=-1)
                if depth_batch.ndim == 4 and depth_batch.shape[1] == 1:
                    depth_batch = np.squeeze(depth_batch, axis=1)
                # Now expect (N,H,W)
                if depth_batch.ndim == 3:
                    depth_t = torch.from_numpy(depth_batch).unsqueeze(1).contiguous()
                else:
                    # attempt to reshape
                    try:
                        depth_t = torch.from_numpy(depth_batch).unsqueeze(1).contiguous()
                    except Exception:
                        depth_t = None

            # Build texts: either a single instruction or replicate per env
            texts = [self.instruction] * self.num_envs

            # Build output dict
            processed = {
                "pixel_values": rgb_t,  # [N,3,H,W]
                "text": texts,
                "proprio": proprio_batch,
            }
            if depth_t is not None:
                processed["depth"] = depth_t  # [N,1,H,W]

            return processed

        else:
            # obs is an ndarray with leading batch dim (N,H,W,C) or (N,C,H,W)
            rgb_batch = self._to_numpy(obs)
            if rgb_batch.ndim == 4 and rgb_batch.shape[1] in (1, 3) and rgb_batch.shape[1] != rgb_batch.shape[3]:
                # (N,C,H,W) -> (N,H,W,C)
                rgb_batch = np.transpose(rgb_batch, (0, 2, 3, 1))

            if rgb_batch.dtype != np.float32:
                if rgb_batch.max() <= 1.0:
                    rgb_batch = rgb_batch.astype(np.float32)
                else:
                    rgb_batch = (rgb_batch.astype(np.float32) / 255.0)

            rgb_t = torch.from_numpy(rgb_batch).permute(0, 3, 1, 2).contiguous()
            processed = {
                "pixel_values": rgb_t,
                "text": [self.instruction] * rgb_t.shape[0],
                "proprio": None,
            }
            return processed

    def _to_numpy(self, x):
        """Convert common image-like types to numpy array."""
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, Image.Image):
            return np.array(x)
        return np.asarray(x)

    def close(self):
        """Close underlying env(s)."""
        try:
            if hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass
