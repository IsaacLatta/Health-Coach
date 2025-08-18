import numpy as np
from typing import Tuple, Dict, Callable, List, Any

from crewai.tools import tool, BaseTool

from health_coach.backend.agents.flows.rl.rl_engine import RLEngine, SimpleQLearningEngine
from health_coach.backend.agents.flows.rl.flows import RLFlow
import health_coach.config as cfg
import health_coach.compare.env as env
import health_coach.rl_data_gen.drift as drift

def reward_function(prev_state: int, current_state: int) -> float:
    res = prev_state - current_state
    return res

def train_agent_worker(
    engine: RLEngine,
    train_ep: List[float],
    state_to_bin_fn: Callable[[float], int] = drift.discretize_probability,
    action_noise_fn: Callable[[int], float] = drift.action_to_noise,
):
    flow = RLFlow()
    flow.set_rl_engine(engine)
    environ = env.AgentQLearningEnvironment(flow, state_to_bin_fn, action_noise_fn)
    environ.run_episode(train_ep)
    return None
