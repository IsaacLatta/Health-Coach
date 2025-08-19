from crewai import Agent

explorer_agent = Agent(
    name="explorer_agent",
    role="Dynamically adjust RL exploration",
    goal=(""),
    backstory=(""),
    verbose=True
)