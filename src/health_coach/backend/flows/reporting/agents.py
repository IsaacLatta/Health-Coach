from crewai import Crew, Process, Task, Agent

from .tools.explanation import generate_prediction_explanation  

# TODO Update the prompts here too this agent needs to explain

explanation_agent = Agent(
        name="explain_agent",
        role="SHAP Explanation Agent.",
        goal="Given patient features and feature names, produce a detailed textual explanation of the model’s output.",
        backstory="You use SHAP to quantify feature contributions.",
        tools=[generate_prediction_explanation],
        verbose=True,
    )
