from abc import abstractmethod, ABC
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class Transition(BaseModel):
    prev_state: int
    curr_state: int
    reward: int
    action: int

class TransitionStore(ABC):
    @abstractmethod
    def get_last(self, id: int) -> Optional[Transition]:
        ...

    @abstractmethod
    def update(self, id: int, transition: Transition):
        ...

class InMemTransitions(TransitionStore):
    def __init__(self):
        self._transitions: Dict[int, List[Transition]] = {}

    def get_last(self, id: int):
        patient_trans = self._transitions.get(id)
        if patient_trans is not None:
            return patient_trans[-1]
        else:
            return None

    def update(self, id: int, transition: Transition):
        patient_trans = self._transitions.get(id)
        if patient_trans is None:
            self._transitions[id] = [transition]
        else:
            patient_trans.append(transition)