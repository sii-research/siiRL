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

import asyncio
from siirl.utils.reward_score.math import compute_score
from typing import Any, Dict, Optional, Tuple
class MathEnv():
    def __init__(self):
        pass
    def reset(self) -> Any:
        pass
    async def step(self, actions, ground_truth):
        actor_action = actions[-1]
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(
            None, 
            compute_score, 
            actor_action, ground_truth 
        )
        should_stop = False
        if score == 1.0:
            next_obs = [act + " This answer is right." for act in actions]
            should_stop = True
        else:
            next_obs = [act + " This answer is wrong." for act in actions]
        return next_obs, score, should_stop
            
    