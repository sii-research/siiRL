from pydantic import BaseModel
from typing import List, Optional, Union

class AgentOutputStatus:
    RUNNING = 0
    LENGTH_FINISH = 1
    ENV_FINISH = 2
    Turn_FINISH = 3

class AgentOutput(BaseModel):
    batch_id: int = -1
    original_prompt: Optional[Union[str, List[int]]]
    response_id: Optional[Union[str, List[int]]]
    templated_prompt: Optional[Union[str, List[int]]]
    should_stop: bool = False
    response_mask: Optional[List[int]]
    env_obs: Optional[Union[str, List[int]]] = ""
    ground_truth: str = ''
    rewards: int = -1
    status: str = AgentOutputStatus.RUNNING
    turn: int = 0
    request_id: str = "None"



     