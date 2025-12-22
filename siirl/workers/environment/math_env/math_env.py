# Copyright (c) 2025, Shanghai Innovation Institute.  All rights reserved.

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

from siirl.workers.environment.base import BaseEnvironment
from typing import Any, Dict, List, Literal, Optional, Tuple

# import dataloder
import random
import numpy as np
import json
from siirl.workers.environment.math_env.parse_utils_qwen import extract_answer as extract_fn, parse_ground_truth
from siirl.workers.environment.math_env.grader import math_equal


def load_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset


def load_profiles(path):
    with open(path, "r") as file:
        profiles = json.load(file)
    return profiles


def extract_answer(answer_str: str) -> str:
    return extract_fn(answer_str, data_name="math")


def extract_ground_truth(ground_truth_str: str) -> str:
    return parse_ground_truth(ground_truth_str, data_name="math")


def judge_correct(extracted_ground_truth: Optional[str], answer: str) -> bool:
    result = math_equal(answer, extracted_ground_truth)
    return result


class MathEnv(BaseEnvironment):
    def __init__(self, env_id, config):
        super().__init__(env_id, config)
        # todo: dataloader load

        profile_path = config["profile_path"]
        dataset_path = config["dataset_path"]
        self.dataset = load_dataset(dataset_path=dataset_path)
        self.profiles = load_profiles(profile_path)
        self.n_agents = config["num_agents"]
        assert self.n_agents == len(self.profiles), "Number of agents must match the number of profiles."
        self.max_steps = config["max_step"]
        self.step_count = 0

        self.problem = None
        self.label = None
        self.current_state = None

    async def reset(self, dp_rank: int, ddp_world_size: int, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        num_data = len(self.dataset) // ddp_world_size
        dataset = self.dataset[dp_rank * num_data : (dp_rank + 1) * num_data]
        problem_answer_pair = random.choice(dataset)
        self.problem = problem_answer_pair["problem"]
        self.label = problem_answer_pair["final_answer"]
        self.current_state = "<|im_start|>problem: " + self.problem + "<|im_end|>\n"
        self.history = []
        obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        self.step_count = 0
        return obs

    async def step(self, actions):
        self.step_count += 1
        actions_to_check = []
        await self.state_transition(actions)
        for i in range(self.n_agents):
            if self.profiles[i]["with_answer"]:
                actions_to_check.append(actions[i])

        score = 0.0
        for action in actions_to_check:
            if self._is_correct(action):
                score += 1.0
        score /= len(actions_to_check)  # normalize

        if score > 0.0 or self.step_count >= self.max_steps:
            dones = np.ones((self.n_agents), dtype=bool)
            # score -= self.step_count # penalize for more steps
        else:
            dones = np.zeros((self.n_agents), dtype=bool)

        if score == 0.0:
            self.current_state = self.current_state + "judge: The answer is incorrect.\n"
        else:
            self.current_state = self.current_state + "judge: The answer is correct.\n"

        next_obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        rewards = [0 if idx != self.n_agents - 1 else score for idx in range(self.n_agents)]
        infos = {"state": self.current_state, "episodic_return": score, "dones": dones}
        return next_obs, rewards, infos

    async def format_observation(self, observation):
        return await super().format_observation(observation)

    async def state_transition(self, actions):
        for i, action in enumerate(actions):
            self.current_state = self.current_state + self.profiles[i]["role"] + ": " + action + "\n"

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        return judge_correct(self.label, extracted_answer)

    def seed(self, seed):
        np.random.seed(seed)

    def get_env_info(self):
        env_info = {"n_agents": self.n_agents}
        return env_info

    async def close(self):
        pass
