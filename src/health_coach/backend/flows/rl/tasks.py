from crewai import Task
from pydantic import BaseModel
from .agents import explorer_agent

class ActionResponse(BaseModel):
    select: int

select_explorer_task = Task(
    name="shape_action",
    description=("You must choose exactly one exploration strategy by its index. Use this mapping:\n"
            "0 : epsilon_greedy\n"
            "1 : softmax\n"
            "2 : ucb\n"
            "3 : thompson_sampling\n"
            "4 : count_bonus\n"
            "5 : maxent\n\n"
            "Return ONLY valid JSON in the form:\n"
            "{ \"select\": <int> }\n"
            "where <int> is an integer from 0 to 5 corresponding to your chosen strategy."),
    agent=explorer_agent,
    output_pydantic=ActionResponse,
    expected_output=("{ \"select\": <int> }")
)