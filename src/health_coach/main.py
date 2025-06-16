#!/usr/bin/env python
import sys
import warnings
import json

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
# from health_coach.tools.data import test_load_patient_history
from health_coach.crew import make_test_pipeline

def test():
    inputs_0 = {
    "patient_info": {"patient_id" : "0" },
    "feature_names" : [ "age","sex","cp","trestbps","chol","fbs", "restecg","thalach","exang","oldpeak","slope","ca","thal" ],
    "features": [ 63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 2 ]
    }
    expected_0 = {
        "Age": 72,
        "Sex": 0,
        "Chest pain type": 4,
        "BP": 130,
        "Cholesterol": 250,
        "FBS over 120": 0,
        "EKG results": 1,
        "Max HR": 140,
        "Exercise angina": 1,
        "ST depression": 1.4,
        "Slope of ST": 2,
        "Number of vessels fluro": 1,
        "Thallium": 3,
        "Heart Disease": 0,
        "Id": 0,
        "Date": "2025-06-10",
        "Prediction": 0.45,
        "State": 4,
        "Action": "DECREASE_MODERATE_THRESHOLD",
        "Reward": -1,
        "Eval_Date": "2025-06-11",
        "Eval_Prediction": 0.38,
        "Next_State": 3,
        "True_Outcome": 0
    }

    crew = make_test_pipeline() 
    try:
        crew_output = crew.kickoff(inputs=inputs_0)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    
    assert crew_output.json_dict["history"] == expected_0, f"Output failed, expected {expected_0}, got {crew_output.json_dict["history"]}"
    print("\n\nTEST PASSED.")
if __name__ == "__main__":
    run()