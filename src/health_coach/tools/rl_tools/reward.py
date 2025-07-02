import math
from typing import Union
from crewai.tools import tool

from health_coach.tools.actions import retrieve_action_index

from health_coach.tools.rl_tools.context import (
    _get_visit_count,
    _get_transition_count,
    _get_episode_length,
)

EXPLORATION_BONUS_FACTOR: float = 0.1 
NOVELTY_BONUS_FACTOR: float = 0.1
TREND_MULTIPLIER: float = 0.9 

@tool
def shape_vcb(state: int, reward: float) -> float:
    """
    VisitationCountBonus: r -> r + EXPLORATION_BONUS_FACTOR / sqrt(N(s) + 1)
    Encourages exploration of under-visited states.
    """
    count = _get_visit_count(state)
    return reward + EXPLORATION_BONUS_FACTOR / math.sqrt(count + 1)

@tool
def shape_rtb(state: int, action: Union[int, str], next_state: int, reward: float) -> float:
    """
    RareTransitionBonus: r -> r + NOVELTY_BONUS_FACTOR / (N(s,a->s') + 1)
    Rewards novel transitions by giving bonus to infrequent (s,a,s') tuples.
    If this reward shaping function is unavailable, the tool returns -1.
    """

    action_index = retrieve_action_index(action)
    if action_index < 0:
        return reward

    count = _get_transition_count(state, action, next_state)
    return reward + NOVELTY_BONUS_FACTOR / (count + 1)

@tool
def shape_trend(reward: float, window: int = 5) -> float:
    """
    TrendShaper: r -> r + TREND_MULTIPLIER * (MA_t - MA_{t-1})
    Boosts the reward when recent reward trend is positive.
    """
    ma = get_moving_avg(window)
    if len(ma) < 2 or math.isnan(ma[-1]) or math.isnan(ma[-2]):
        return reward
    delta = ma[-1] - ma[-2]
    return reward + TREND_MULTIPLIER * delta

def _amplify_reward(reward: float, factor: float) -> float:
    return reward * factor

@tool
def shape_amplify(pure_reward: float, factor: float) -> float:
    """
    Amplify the given pure rl reward by the specified factor.
    """
    return _amplify_reward(pure_reward, factor)

@tool
def shape_progress(reward: float, timestep: int) -> float:
    """
    EpisodeProgressShaper: r -> r * (timestep / L)
    Gradually increases reward influence as episode progresses.
    """
    L = _get_episode_length()
    if L <= 0:
        return reward
    return reward * (timestep / L)

def get_all_tools():
    return [shape_vcb, shape_rtb, shape_trend, shape_progress, shape_amplify]
