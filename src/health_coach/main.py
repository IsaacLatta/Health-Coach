#!/usr/bin/env python
import sys
import warnings

from health_coach.crew import make_health_coach

from datetime import datetime

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    inputs = {
        "feature_names" : [ "age","sex","cp","trestbps","chol","fbs", "restecg","thalach","exang","oldpeak","slope","ca","thal" ],
        "features": [ 63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 2 ]
    }

    try:
        make_health_coach().kickoff(inputs=inputs)
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

from health_coach.tools.data import test_load_config
from health_coach.tools.data import test_load_patient_history

def test():
    # test_load_config()
    test_load_patient_history()
    return
    # inputs = {
    #     "topic": "AI LLMs",
    #     "current_year": str(datetime.now().year)
    # }
    
    # try:
    #     HealthCoach().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    # except Exception as e:
    #     raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    run()