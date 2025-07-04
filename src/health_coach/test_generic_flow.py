# flows.py

from typing import TypeVar, Generic, Optional
from pydantic import BaseModel
from pydantic.generics import GenericModel
from crewai.flow.flow import Flow, start, listen
from rl import RLEngine

StateType  = TypeVar("StateType")
ActionType = TypeVar("ActionType")
RewardType = TypeVar("RewardType")

class RLInput(BaseModel):
    patient_id: str
    patient_features: list[float]

class RLState(GenericModel, Generic[StateType, ActionType, RewardType]):
    input: Optional[RLInput] = None
    prev_state: Optional[StateType] = None
    cur_state: Optional[StateType] = None
    env_reward: Optional[RewardType] = None
    context: Optional[str] = None
    action: Optional[ActionType] = None
    shaped_reward: Optional[RewardType]  = None

class RLFlow(Flow[RLState[StateType, ActionType, RewardType]], Generic[StateType, ActionType, RewardType]):
    def set_input(self, input: RLInput):
        self.state.input = input

    def set_rl_engine(self, engine: RLEngine[StateType, ActionType, RewardType]):
        self.engine = engine

    @start()
    def start_stage(self):
        if self.state.input is None:
            raise RuntimeError("Must call set_input(...) before kickoff()")
        self.state.prev_state = self.engine.encode_state(self.state.input)

    @listen(start_stage)
    def step(self):
        self.state.cur_state = self.engine.encode_state(self.state.input)

        self.state.env_reward = self.engine.compute_env_reward(self.state.prev_state, self.state.cur_state)

        self.state.context = self.engine.generate_context(self.state.prev_state, self.state.cur_state, self.state.env_reward)

        self.state.action, self.state.shaped_reward = self.engine.shape_reward_and_action(
            self.state.prev_state, 
            self.state.cur_state, 
            self.state.env_reward, 
            self.state.context
            )

        _ = self.engine.update_model(self.state.prev_state, self.state.action, self.state.shaped_reward, self.state.cur_state)

        return {"action": self.state.shaped_reward, "shaped_reward": self.state.shaped_reward}
