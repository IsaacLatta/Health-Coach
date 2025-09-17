from pydantic import BaseModel
from typing import Optional, Any
import numpy as np
from crewai.flow.flow import Flow, start, listen, and_
from crewai.flow.persistence import persist
from health_coach.compare.rl import RLEngine, QLearningState

class RLState(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    rl_engine: Optional[RLEngine] = None
    prev_state: int = 0
    curr_state: int = 0
    env_reward: float = 0.0
    context: str = ""
    action: int = 1
    shaped_reward: float = 0.0
    input: Any = None

class RLFlow(Flow[RLState]):

    def set_rl_engine(self, engine: RLEngine):
        self.state.rl_engine = engine

    def set_state(self, previous_state, current_state):
        self.state.prev_state = previous_state
        self.state.curr_state = current_state

    @start()
    def compute_reward(self):
        self.state.env_reward = self.state.rl_engine.compute_env_reward(self.state.prev_state, self.state.curr_state)

    @listen(compute_reward)
    def make_context(self):
        eng = self.state.rl_engine
        self.state.context = eng.generate_context(
            self.state.prev_state,
            self.state.curr_state,
            self.state.env_reward
        )
        return self.state.context

    @listen(make_context)
    def shape_action(self):
        self.state.action = self.state.rl_engine.shape_action(
            self.state.prev_state,
            self.state.curr_state,
            self.state.env_reward,
            self.state.context
        )

    @listen(make_context)
    def shape_reward(self):
        self.state.shaped_reward = self.state.rl_engine.shape_reward(
            self.state.prev_state,
            self.state.curr_state,
            self.state.env_reward,
            self.state.context
        )
    
    @listen(and_(shape_reward, shape_action))
    def update_model(self):
        self.state.rl_engine.save_state(
            self.state.prev_state, 
            self.state.action, 
            self.state.shaped_reward, 
            self.state.curr_state)
        
        return self.state.action

