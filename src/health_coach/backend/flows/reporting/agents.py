from crewai import Agent

from health_coach.backend.flows.llm import get_embedder, get_llm

shap_explainer_agent = Agent(
    name="shap_explainer",
    role="Feature Attribution Explainer",
    goal=(
        "Given a list of SHAP items for a single patient, write a short, "
        "patient-friendly explanation for each feature’s contribution, "
        "without changing any numbers."
    ),
    backstory=(
        "You are an explainability specialist. You translate SHAP values into "
        "clear, non-technical language for clinicians. You never provide medical advice; "
        "you only explain model behavior and contributing factors."
    ),
    verbose=True,
    #llm=llm.get_llm(),
    #embedder=llm.get_embedder()
)

risk_summary_agent = Agent(
    name="risk_summarizer",
    role="Risk Profile Summarizer",
    goal=(
        "Summarize the patient’s overall risk and why, using the probability, risk label, "
        "thresholds, and the top feature contributions. Keep it precise, neutral, and brief."
    ),
    backstory=(
        "You are a clinical-facing analyst. You synthesize model outputs into a brief, "
        "readable risk overview. You avoid recommendations or diagnoses and include a "
        "clear caveat that this is for decision support only."
    ),
    tools=[],
    verbose=True,
    #llm=llm.get_llm(),
    #embedder=llm.get_embedder()
)
