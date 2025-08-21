from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from crewai import Task
from .agents import insights_agent

class ActionStat(BaseModel):
    action: int
    count: int
    mean_reward: float

class Metrics(BaseModel):
    probability: float
    thresholds: Dict[str, float]
    margin_to_moderate: float    
    margin_to_high: float        
    volatility: Optional[Literal["low","medium","high"]] = None
    volatility_value: Optional[float] = None
    config_deltas: Dict[str, float] = Field(default_factory=dict)
    action_stats: List[ActionStat] = Field(default_factory=list) 
    notes: Optional[str] = None

class InsightsSummary(BaseModel):
    narrative: str
    bullets: List[str] = Field(default_factory=list)

insights_task = Task(
    name="rl_insights_task",
    description=(
        "You are given RL metrics for a single patient:\n"
        "{metrics_json}\n\n"
        "Write a short, clinician-friendly explanation (no medical advice):\n"
        "1) What the system observed (margins to thresholds, volatility).\n"
        "2) How configuration changed vs defaults and why that might help.\n"
        "3) Which actions tended to reduce risk (positive reward).\n"
        "Keep 3–5 sentences total, neutral tone. Also provide 3–5 concise bullets."
    ),
    agent=insights_agent,
    output_pydantic=InsightsSummary,
    expected_output="{\"narrative\": str, \"bullets\": List[str]}"
)
