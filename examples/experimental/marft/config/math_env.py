from siirl.workers.environment.base import BaseEnvironment
from siirl.utils.reward_score.math import compute_score
from typing import Any, Dict, Optional, Tuple
class MathEnv(BaseEnvironment):
    def __init__(self):
        pass
    def reset(self) -> Any:
        pass
    def step(self, actions, ground_truth):
        actor_action = actions[-1]
        score = compute_score(actor_action, ground_truth)
        should_stop = False
        if score == 1.0:
            next_obs = [act + ". This answer is right." for act in actions]
            should_stop = True
        else:
            next_obs = [act + ". This answer is wrong." for act in actions]
        return next_obs, score, should_stop
            
    