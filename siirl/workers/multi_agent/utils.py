from pydantic import BaseModel
from typing import List, Optional, Union

class AgentOutputStatus:
    RUNNING = 0
    LENGTH_FINISH = 1

class AgentOutput(BaseModel):
    batch_id: int = -1
    original_prompt: Optional[Union[str, List[int]]]
    response_id: Optional[Union[str, List[int]]]
    templated_prompt: Optional[Union[str, List[int]]]
    should_stop: bool = False
    response_mask: Optional[List[int]]
    status: str = AgentOutputStatus.RUNNING
    turn: int = 0
    request_id: str = "None"



     