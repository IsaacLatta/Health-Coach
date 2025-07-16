#!/usr/bin/env python
import sys
import warnings
import json

# from health_coach.crew import make_health_coach
import health_coach.flows as my_flows

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

def train():
    print("Not implemented.")
    return
    # inputs = {
    #     "topic": "AI LLMs",
    #     'current_year': str(datetime.now().year)
    # }
    # try:
    #     HealthCoach().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    # except Exception as e:
    #     raise Exception(f"An error occurred while training the crew: {e}")

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
from health_coach.flows import RLFlow, RLInput
def test():
    flow = RLFlow()
    
    flow.set_input(RLInput(
        patient_id="0",
        patient_features= [41.0, 1.0, 4.0, 110.0, 172.0, 0.0, 2.0, 158.0, 0.0, 0.0, 1.0, 0.0, 7.0]
    ))

    agent_engine = QLearningEngine(
            exploration_tools=rl_tools.action.get_all_tools(),
            context_tools=rl_tools.context.get_all_tools(),
            shaping_tools=rl_tools.reward.get_all_tools(),
            max_iter=3,
        )

    flow.set_rl_implementation(agent_engine)


    result = flow.kickoff()

    # next_flow = RLFlow()
    
    # next_flow.set_input(RLInput(
    #     patient_id="0",
    #     patient_features= [41.0, 1.0, 4.0, 110.0, 172.0, 0.0, 2.0, 158.0, 0.0, 0.0, 1.0, 0.0, 7.0]
    # ))

    # next_flow.set_rl_implementation(
    #     QLearningEngine(
    #         exploration_tools=rl_tools.action.get_all_tools(),
    #         context_tools=rl_tools.context.get_all_tools(),
    #         shaping_tools=rl_tools.reward.get_all_tools(),
    #         use_crews=True,
    #         max_iter=3,
    #     )
    # )

    # next_result = next_flow.kickoff()
    
    # flow = PersistentCounterFlow()
    # result = flow.kickoff(
    #     inputs={
    #         'id': 'my-unique-id'
    #     }
    # )
    # print(f"= This run result: {result}")

    # next_flow = PersistentCounterFlow()
    # result = next_flow.kickoff(
    #     inputs={
    #         'id': 'my-unique-id'
    #     }
    # )
    # print(f"= This next run result: {result}")

    
if __name__ == "__main__":
    run()