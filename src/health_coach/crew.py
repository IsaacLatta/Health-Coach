from crewai import Agent
from crewai.project import CrewBase, agent, crew, task
from crewai import Crew, Process, Task  

from health_coach.tools.prediction import predict_heart_disease
from health_coach.tools.reporting import generate_report, save_report
from health_coach.tools.explanation import generate_prediction_explanation

from health_coach.agents import data_input_agent
from health_coach.tasks import get_load_configuration_task
from health_coach.tasks import get_fetch_patient_history_task

def make_test_pipeline():
    agent = data_input_agent
    task = get_fetch_patient_history_task(agent)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

def make_health_coach():
    prediction_agent = Agent(
        name="prediction_agent",
        role=(
            "Prediction Agent for cardiovascular risk assessment "
            "in clinical settings."
        ),
        goal=(
            "Given a 13-element feature vector (age, BP, chol, etc.), "
            "compute and return the probability of heart disease."
        ),
        backstory=(
            "You are a medical-AI specialist. You know how to load "
            "a pickled scikit-learn logistic-regression model and apply it "
            "to a numerical feature array."
        ),
        tools=[predict_heart_disease],
        verbose=True,
    )

    prediction_task = Task(
        name="prediction_task",
        description=(
            "Run the Prediction Agent on input patient features {features} "
            "to get a risk probability."
        ),
        agent=prediction_agent,            
        tools=[predict_heart_disease],     
        expected_output='{"probability": float}',
    )

    explanation_agent = Agent(
        name="explain_agent",
        role="SHAP Explanation Agent.",
        goal="Given patient features, produce a detailed HTML explanation of the model’s output.",
        backstory="You use SHAP to quantify feature contributions.",
        tools=[generate_prediction_explanation],
        verbose=True,
    )

    explanation_task = Task(
        name="explanation_task",
        description="Explain the prediction for features {features} with feature names {feature_names}.",
        agent=explanation_agent,
        tools=[generate_prediction_explanation],
        tools_input={
            "generate_prediction_explanation": {
                "features": "{features}",
                "feature_names": "{feature_names}"
            }
        },
        expected_output='{"html": str}',
    )

    report_agent = Agent(
        name="report_agent",
        role="Report Generation Agent.",
        goal="Given features, probability, and brief explanation, produce a patient report in HTML.",
        backstory="You know how to fill an HTML medical-report template.",
        tools=[generate_report],
        verbose=True,
    )

    report_task = Task(
        name="report_task",
        description=(
          "Generate an HTML report for features {features} "
          "and heart disease prediction. ALWAYS save the report file to disk after generation."
        ),
        agent=report_agent,
        tools=[generate_report],
        tools_input={ 
            "generate_report": {
                "features":   "{features}",
                "probability":"{prediction_task.probability}",
                "explanation":"{explanation_task.html}"
            }
        },
        expected_output='{"html": str}',
    )

    save_task = Task(
        name="save_report_task",
        description="Save the generated HTML to disk.",
        agent=report_agent,
        tools=[save_report],
        tools_input={
            "save_report": {
                "html": "{report_task.html}"
            }
        },
        expected_output='{"success": bool}',
    ) 

    return Crew(
        agents=[prediction_agent, explanation_agent, report_agent],
        tasks=[prediction_task, explanation_task, report_task, save_task],
        process=Process.sequential,
        verbose=True,
    )

