from pydantic import BaseModel

from crewai import Crew, Process, Task, Agent

from .agents import explanation_agent
from .tools.explanation import generate_prediction_explanation  

# TODO: Update the prompts

class ShapExplanationResponse(BaseModel):
    ...

# UPDATE TO: map of shap values to explanations
explanation_task = Task(
        name="explanation_task",
        description="Explain the prediction for features {features} with feature names {feature_names}.",
        agent=explanation_agent,
        #expected_output='{"html": str}', # Update this with the new format
        output_json= ShapExplanationResponse,
        output_pydantic=ShapExplanationResponse
    )

# UPDATE TO: an overall explanation
report_task = Task(
    name="report_task",
    description=(
        "Generate an HTML report for features {features} "
        "and heart disease prediction. ALWAYS save the report file to disk after generation."
    ),
    agent=explanation_agent,
    tools_input={ 
        "generate_report": {
            "features":   "{features}",
            "probability":"{prediction_task.probability}",
            "explanation":"{explanation_task.html}"
        }
    },
    expected_output='{"html": str}',
)