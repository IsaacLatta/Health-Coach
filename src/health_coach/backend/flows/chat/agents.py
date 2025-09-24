from crewai import Agent

from health_coach.backend.flows.llm import get_embedder, get_llm 

chat_agent = Agent(
    role="Clinician-facing explainer",
    goal=(
        "Answer the clinician's question about this patient using ONLY the provided "
        "context (risk report, SHAP drivers, RL metrics, config deltas, last transitions). "
        "Be precise, neutral, non-prescriptive; avoid medical advice. Cite which signals you used."
    ),
    backstory=(
        "You are assisting clinicians by explaining model predictions and the reinforcement-"
        "learning configuration behavior for a single patient. You never recommend treatment."
    ),
    verbose=False,
    #llm=llm.get_llm(),
    #embedder=llm.get_embedder()
)
