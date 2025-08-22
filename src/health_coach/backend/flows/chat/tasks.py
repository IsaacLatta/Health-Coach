from typing import List, Optional
from pydantic import BaseModel, Field
from crewai import Task
from .agents import chat_agent

class ChatReply(BaseModel):
    answer: str
    bullets: List[str] = Field(default_factory=list)
    used_signals: List[str] = Field(default_factory=list)
    safety_note: Optional[str] = "Decision support only; not medical advice."

chat_task = Task(
    name="clinician_chat",
    description=(
        "Context JSON (single patient): {context_json}\n"
        "Clinician question: {question}\n\n"
        "Answer concisely (3–6 sentences). Then provide 2–5 concise bullets. "
        "Clearly list which signals you used (e.g., probability, thresholds, SHAP, RL insights). "
        "Do NOT give medical advice."
    ),
    agent=chat_agent,
    output_pydantic=ChatReply,
    expected_output='{"answer":"<string>", "bullets":["<string>"], "used_signals":["<string>"], "safety_note":"<string>"}'
)
