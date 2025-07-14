import itertools

from health_coach.flows import RLFlow
from health_coach.rl import QLearningEngine, RLEngine
import health_coach.tools.rl_tools.action as actions
import health_coach.tools.rl_tools.context as context
import health_coach.tools.rl_tools.reward as reward


def run_pure():
    # pure_actions    = actions.get_all_tool_funcs()
    # pure_rewards    = reward.get_all_tool_funcs()
    return

def run_flow(engine: RLEngine):
    ...

def run_agent_rl():
    action_tools   = actions.get_all_tools()
    context_tools  = context.get_all_tools()
    reward_tools   = reward.get_all_tools()
    tools = ((action_tools, context_tools, reward_tools))
    for (action, context, reward) in itertools.product(*tools):
        engine = QLearningEngine(exploration_tools=action, context_tools=context, shaping_tools=reward)
        run_flow(engine)

def run_comparison():
    for use_crews in (True, False):
        if use_crews:
            run_agent_rl()
        else:
            run_pure()


