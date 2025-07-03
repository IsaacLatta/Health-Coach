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

import health_coach.tests.integration.rl as rl_tests
from health_coach.rl import QLearningImplementation

def test():
    flow = my_flows.RLFlow()
    input = my_flows.RLInput()
    input.patient_id = "0"
    input.patient_features = [41.0, 1.0, 4.0, 110.0, 172.0, 0.0, 2.0, 158.0, 0.0, 0.0, 1.0, 0.0, 7.0]

    flow.set_input(input)
    flow.set_rl_implementation(QLearningImplementation())
    result = flow.kickoff()
    print(f"Result is {result}")
    # rl_tests.test_all()

if __name__ == "__main__":
    run()