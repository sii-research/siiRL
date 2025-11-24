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

import unittest
import numpy as np
import torch
import gymnasium as gym
import os
import warnings

# Ignore robosuite/sapien deprecation warnings (e.g., pkg_resources, logger.warn)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="robosuite")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sapien")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mani_skill")
# Catch specific logger warnings that don't always respect category filters
warnings.filterwarnings("ignore", message=".*The 'warn' method is deprecated.*")
# Ignore Vulkan warnings (rendering driver issues on headless servers)
warnings.filterwarnings("ignore", message=".*Failed to find system libvulkan.*")
warnings.filterwarnings("ignore", message=".*Failed to find Vulkan ICD file.*")
# Ignore NumPy 2.0 specific warnings
warnings.filterwarnings("ignore", message=".*__array_wrap__ must accept context.*")
warnings.filterwarnings("ignore", message=".*env.single_action_space to get variables.*")

from siirl.workers.environment.embodied import ManiSkillAdapter

class TestManiSkillAdapter(unittest.TestCase):
    
    def setUp(self):
        """ Setup test environment. """
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
             os.environ["CUDA_VISIBLE_DEVICES"] = "0"
             
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"\n[Test Setup] Running tests on device: {self.device}")

    def test_init_native_vector(self):
        """Test native vectorization initialization"""
        num_envs = 2
        env = ManiSkillAdapter("PickCube-v1", num_envs=num_envs)
        self.assertTrue(env.is_vector_env)
        self.assertEqual(env.num_envs, num_envs)
        # ManiSkill 3 uses its own vectorizer, not Gym's SyncVectorEnv
        self.assertNotIsInstance(env.env, gym.vector.SyncVectorEnv)
        env.close()

    def test_single_env_lifecycle(self):
        """Test single environment lifecycle"""
        env = ManiSkillAdapter("PickCube-v1", num_envs=1)
        obs, _ = env.reset()
        pixel_values = obs["pixel_values"]
        self.assertEqual(pixel_values.shape[0], 1, "Should maintain batch dim 1 even for single env")
        self.assertEqual(pixel_values.ndim, 4, "Shape should be [1, C, H, W]")
        proprio = obs["proprio"]
        self.assertEqual(proprio.shape[0], 1)
        self.assertEqual(proprio.ndim, 2, "Shape should be [1, Dim]")
        env.close()

    def test_reset_data_location_and_shape(self):
        """Test that data stays on the correct device (GPU) and has correct VLA shape"""
        num_envs = 2
        env = ManiSkillAdapter("PickCube-v1", num_envs=num_envs)
        processed_obs, info = env.reset()
        pixel_values = processed_obs["pixel_values"]
        self.assertIsInstance(pixel_values, torch.Tensor)
        self.assertEqual(pixel_values.device.type, self.device.type)
        self.assertEqual(pixel_values.ndim, 4) 
        self.assertEqual(pixel_values.shape[0], num_envs)
        self.assertEqual(pixel_values.shape[1], 3) # RGB
        proprio = processed_obs["proprio"]
        self.assertIsInstance(proprio, torch.Tensor)
        self.assertEqual(proprio.device.type, self.device.type)
        self.assertEqual(proprio.ndim, 2) 
        self.assertEqual(proprio.shape[0], num_envs)
        self.assertIsInstance(processed_obs["text"], list)
        self.assertEqual(len(processed_obs["text"]), num_envs)
        self.assertIsInstance(processed_obs["text"][0], str)
        env.close()

    def test_rgb_normalization(self):
        """Verify RGB images are float32 in range [0, 1], not uint8 [0, 255]"""
        env = ManiSkillAdapter("PickCube-v1", num_envs=1)
        obs, _ = env.reset()
        pixels = obs["pixel_values"]
        self.assertEqual(pixels.dtype, torch.float32)
        self.assertTrue(pixels.max() <= 1.0 + 1e-5, f"Max pixel value {pixels.max()} > 1.0")
        self.assertTrue(pixels.min() >= 0.0, f"Min pixel value {pixels.min()} < 0.0")
        env.close()

    def test_depth_channel(self):
        """Verify Depth channel is processed correctly if available"""
        # PickCube-v1 usually supports RGBD by default with the args provided
        env = ManiSkillAdapter("PickCube-v1", num_envs=2)
        obs, _ = env.reset()
        if "depth" in obs:
            depth = obs["depth"]
            self.assertIsInstance(depth, torch.Tensor)
            self.assertEqual(depth.device.type, self.device.type)
            self.assertEqual(depth.ndim, 4)
            self.assertEqual(depth.shape[1], 1, "Depth should have 1 channel")
            self.assertEqual(depth.dtype, torch.float32)
        else:
            print("[Warning] Depth not found in env, skipping depth checks.")
        env.close()

    def test_step_tensor_flow_and_rewards(self):
        """Test stepping with tensors and verifying return types"""
        num_envs = 2
        env = ManiSkillAdapter("PickCube-v1", num_envs=num_envs)
        env.reset()
        
        try:
            sas = env.env.get_wrapper_attr("single_action_space")
            action_dim = sas.shape[0]
        except AttributeError:
            action_dim = env.env.action_space.shape[-1]
             
        action_cpu = torch.zeros((num_envs, action_dim), dtype=torch.float32)
        processed_obs, reward, terminated, truncated, info = env.step(action_cpu)
        self.assertIsInstance(reward, torch.Tensor)
        self.assertEqual(reward.device.type, self.device.type)
        self.assertEqual(reward.shape[0], num_envs)
        self.assertTrue(reward.ndim in [1, 2], "Reward should be 1D [N] or 2D [N, 1]")
        self.assertIsInstance(terminated, torch.Tensor)
        self.assertIsInstance(truncated, torch.Tensor)
        env.close()

    def test_flexible_action_inputs(self):
        """Test adapter accepts List, Numpy, and Tensor actions"""
        num_envs = 1
        env = ManiSkillAdapter("PickCube-v1", num_envs=num_envs)
        env.reset()
        
        try:
            sas = env.env.get_wrapper_attr("single_action_space")
            action_dim = sas.shape[0]
        except AttributeError:
            action_dim = env.env.action_space.shape[-1]

        list_action = [0.0] * action_dim
        _, r_list, _, _, _ = env.step(list_action)
        self.assertIsInstance(r_list, torch.Tensor)
        np_action = np.zeros((1, action_dim), dtype=np.float32)
        _, r_np, _, _, _ = env.step(np_action)
        self.assertIsInstance(r_np, torch.Tensor)
        if torch.cuda.is_available():
            gpu_action = torch.zeros((1, action_dim), device="cuda")
            _, r_gpu, _, _, _ = env.step(gpu_action)
            self.assertIsInstance(r_gpu, torch.Tensor)
        env.close()

    def test_physics_stepped(self):
        """Verify that the environment state actually changes after a step"""
        num_envs = 1
        env = ManiSkillAdapter("PickCube-v1", num_envs=num_envs)
        obs_start, _ = env.reset()
        start_proprio = obs_start["proprio"].clone()
        
        try:
            sas = env.env.get_wrapper_attr("single_action_space")
            action_dim = sas.shape[0]
        except AttributeError:
            action_dim = env.env.action_space.shape[-1]
             
        action = torch.rand((num_envs, action_dim), device=self.device) * 2.0 - 1.0
        obs_next, _, _, _, _ = env.step(action)
        next_proprio = obs_next["proprio"]
        diff = (next_proprio - start_proprio).abs().sum()
        self.assertGreater(diff.item(), 0.0, "Environment state did not change after step (Proprio is identical)")
        env.close()

    def test_broadcasting(self):
        """Test that single action broadcasts to multiple envs internally"""
        num_envs = 3
        env = ManiSkillAdapter("PickCube-v1", num_envs=num_envs)
        env.reset()
        
        try:
            sas = env.env.get_wrapper_attr("single_action_space")
            action_dim = sas.shape[0]
        except AttributeError:
            action_dim = env.env.action_space.shape[-1]

        single_action = torch.zeros(action_dim, dtype=torch.float32)
        processed_obs, reward, _, _, _ = env.step(single_action)
        self.assertEqual(reward.shape[0], num_envs)
        env.close()

if __name__ == "__main__":
    unittest.main()
