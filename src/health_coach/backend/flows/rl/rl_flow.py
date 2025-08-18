from pydantic import BaseModel
from typing import Optional, Any
import numpy as np
from crewai.flow.flow import Flow, start, listen, and_
from crewai.flow.persistence import persist
from .rl_engine import SimpleQLearningEngine, RLEngine
from .tools.exploration import get_all_funcs
from .dependencies import RLDeps

import health_coach.config as cfg

class RLState(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    dependencies: Optional[RLDeps] = None
    rl_engine: Optional[RLEngine] = None
    prev_state: int = 0
    curr_state: int = 0
    env_reward: float = 0.0
    context: str = ""
    action: int = 1
    shaped_reward: float = 0.0
    input: Any = None

class RLFlow(Flow[RLState]):

    def __init__(self, dependencies, inputs, prev_state, cur_state):
        self.state.dependencies = dependencies
        self.state.rl_engine = self.state.dependencies.engine # for convenience
        self.state.inputs = inputs
        self.state.prev_state = prev_state
        self.curr_state = cur_state

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
    

def call_rl_flow(inputs):
    # this func isnt async, it kicks off the task in the bg,  
    # returns immediatly

    # here we can also make a prediction
    # pull down the patient's last prediction
    # or do it inside the flow

    def reward_function(prev_state: int, current_state: int) -> float:
        return prev_state - current_state

    engine = SimpleQLearningEngine(
        get_all_funcs(), 
        reward_function,
        cfg.ALPHA,
        cfg.GAMMA
        )
        
    prev_state = cur_state = 1234
    flow = RLFlow(engine, inputs, prev_state, cur_state)
    result = flow.kickoff()
    return result