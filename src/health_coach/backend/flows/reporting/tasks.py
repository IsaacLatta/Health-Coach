from typing import Any, List, Literal, Optional, Dict
from pydantic import BaseModel, Field
from crewai import Task
from .agents import shap_explainer_agent, risk_summary_agent

class ShapExplanationItem(BaseModel):
    feature: str
    value: Any
    phi: float
    direction: Literal["increases","decreases","mixed"]
    magnitude: float
    explanation: str

class ShapExplanations(BaseModel):
    items: List[ShapExplanationItem] = Field(default_factory=list)
    notes: Optional[str] = Field(
        default="Explanations reflect model attributions, not medical advice."
    )

class RiskSummary(BaseModel):
    risk_label: Literal["low","medium","high"]
    probability: float
    thresholds: Dict[str, float]
    key_drivers: List[str]
    narrative: str
    caveats: List[str] = Field(
        default_factory=lambda: [
            "Model output is supportive information only and not a diagnosis."
        ]
    )

shap_explanations_task = Task(
    name="shap_explanations_task",
    description=(
        "You are given a list of SHAP items for a single patient.\n"
        "Data: {shap}\n"
        "Each item has: feature, value, phi (may be positive or negative).\n\n"
        "For EACH item:\n"
        " - Set direction to 'increases' if phi>0, 'decreases' if phi<0, else 'mixed'.\n"
        " - Set magnitude to abs(phi).\n"
        " - Write one short, patient-friendly explanation of HOW this feature influenced the risk.\n"
        "Do NOT change any numeric values. Keep explanations neutral and non-clinical."
    ),
    agent=shap_explainer_agent,
    output_pydantic=ShapExplanations,
    expected_output=(
        "Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "items": [\n'
        '    {"feature": "str", "value": <any>, "phi": 0.0, '
        '"direction": "increases|decreases|mixed", "magnitude": 0.0, "explanation": "str"}\n'
        "  ],\n"
        '  "notes": "str (optional)"\n'
        "}"
    ),
)

risk_summary_task = Task(
    name="risk_summary_task",
    description=(
        "Summarize overall risk for a single patient.\n"
        "Inputs:\n"
        " - probability: {probability}\n"
        " - risk_label: {risk_label}\n"
        " - thresholds: {thresholds}\n"
        " - shap_explanations: {shap_explanations_task.output.pydantic}\n\n"
        "Requirements:\n"
        " - Provide a concise narrative (3–6 sentences) explaining WHY the model assigned this risk.\n"
        " - List key drivers, ordered by absolute impact (top-first), using feature names only.\n"
        " - Include a brief caveat that this is decision support, not medical advice.\n"
        " - Keep tone clinical-neutral and avoid recommendations or treatment."
    ),
    agent=risk_summary_agent,
    output_pydantic=RiskSummary,
    expected_output=(
        "Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "risk_label": "low|medium|high",\n'
        '  "probability": 0.0,\n'
        '  "thresholds": {"low": 0.0, "medium": 0.0, "high": 0.0},\n'
        '  "key_drivers": ["str", "..."],\n'
        '  "narrative": "str",\n'
        '  "caveats": ["str", "..."]\n'
        "}"
    ),
)
