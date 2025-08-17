#!/usr/bin/env python
import sys
import warnings
import json
import time

# from health_coach.crew import make_health_coach

from datetime import datetime

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    inputs = {
        "feature_names" : [ "age","sex","cp","trestbps","chol","fbs", "restecg","thalach","exang","oldpeak","slope","ca","thal" ],
        "features": [ 63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 2 ]
    }

    try:
        # make_health_coach().kickoff(inputs=inputs)
        pass
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

from health_coach.compare.main_compare import run_agent

def train():
    run_agent()

    return
    

def replay():
    print("Not implemented.")
    return
    # try:
    #     HealthCoach().crew().replay(task_id=sys.argv[1])

    # except Exception as e:
    #     raise Exception(f"An error occurred while replaying the crew: {e}")

import health_coach.tools.rl_tools as rl_tools 
import health_coach.tools.rl_tools.action
import health_coach.tools.rl_tools.context
import health_coach.tools.rl_tools.reward

from crewai.utilities.paths import db_storage_path
from pathlib import Path

from health_coach.rl import  QLearningEngine
from health_coach.flows import RLFlow, StateExampleFlow 

from health_coach.compare.main_compare import run_pure, run_agent

import health_coach.config as cfg
from health_coach.rl import SimpleQLearningEngine
from health_coach.compare.train import reward_function
from health_coach.tools.rl_tools.action import get_all_tools

def test():
    # run_agent()

    start_time = time.time()

    engine = SimpleQLearningEngine(
        get_all_tools, 
        reward_function, 
        cfg.ALPHA, 
        cfg.GAMMA
    )

    flow = RLFlow()
    flow.set_rl_engine(engine)
    flow.set_state(1, 2)
    res = flow.kickoff()
    # print(f"Got {res}, state: {flow.state}")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total execution time: {elapsed:.2f} seconds for model {cfg.LLM_MODEL}")

    # flow = StateExampleFlow()
    # # flow.plot("my_flow_plot")
    # final_output = flow.kickoff()
    # print(f"Final Output: {final_output}")
    # print("Final State:")
    # print(flow.state)


    # run_comparison()
    # flow = RLFlow()
    
    # flow.set_input(RLInput(
    #     patient_id="0",
    #     patient_features= [41.0, 1.0, 4.0, 110.0, 172.0, 0.0, 2.0, 158.0, 0.0, 0.0, 1.0, 0.0, 7.0]
    # ))

    # agent_engine = QLearningEngine(
    #         exploration_tools=rl_tools.action.get_all_tools(),
    #         context_tools=rl_tools.context.get_all_tools(),
    #         shaping_tools=rl_tools.reward.get_all_tools(),
    #         max_iter=3,
    #     )

    # flow.set_rl_engine(agent_engine)


    # result = flow.kickoff()
    return

    
