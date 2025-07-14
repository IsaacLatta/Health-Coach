from pydantic import BaseModel
from typing import Optional
import numpy as np
from crewai.flow.flow import Flow, start, listen, and_
from crewai.flow.persistence import persist
from health_coach.rl import RLEngine, QLearningState

class RLInput(BaseModel):
    patient_id: str
    patient_features: list[float]

@persist()
class RLState(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    q_state: Optional[QLearningState] = None
    rl_engine: Optional[RLEngine] = None
    input: Optional[RLInput] = None
    prev_state: int = 0
    curr_state: int = 0
    env_reward: float = 0.0
    context: str = ""
    action: int = 0
    shaped_reward: float = 0.0

class RLFlow(Flow[RLState]):
    def set_input(self, input: RLInput):
        self.state.input = input

    def set_rl_implementation(self, engine: RLEngine):
        self.state.rl_engine = engine
        if self.state.input is None:
            raise RuntimeError("Must call set_input() first")
        self.state.prev_state = engine.encode_prev_state(self.state.input)

    @start()
    def encode_previous_state(self):
        eng = self.state.rl_engine
        self.state.prev_state = eng.encode_prev_state(self.state.input)
        return

    @listen(encode_previous_state)
    def encode_state(self):
        eng = self.state.rl_engine
        self.state.curr_state = eng.encode_curr_state(self.state.input)
        return

    @listen(encode_state)
    def compute_reward(self):
        eng = self.state.rl_engine
        self.state.env_reward = eng.compute_env_reward(self.state.prev_state, self.state.curr_state)
        return

    @listen(compute_reward)
    def make_context(self):
        eng = self.state.rl_engine
        self.state.context = eng.generate_context(
            self.state.prev_state,
            self.state.curr_state,
            self.state.env_reward
        )
        return

    @listen(make_context)
    def shape_action(self):
        eng = self.state.rl_engine
        self.state.action = eng.shape_action(
            self.state.prev_state,
            self.state.curr_state,
            self.state.env_reward,
            self.state.context
        )
        print(f"Shaped action {self.state.action}")
        return

    # @listen(make_context)
    # def shape_reward(self):
    #     self.state.shaped_reward = self.state.rl_engine.shape_reward(
    #         self.state.prev_state,
    #         self.state.curr_state,
    #         self.state.env_reward,
    #         self.state.context
    #     )
    #     return
    
    # @listen(and_(shape_reward, shape_action))
    # def update_model(self):
    #     self.state.q_state = self.state.rl_engine.save_state(
    #         self.state.prev_state, 
    #         self.state.action, 
    #         self.state.shaped_reward, 
    #         self.state.curr_state)
    #     return
