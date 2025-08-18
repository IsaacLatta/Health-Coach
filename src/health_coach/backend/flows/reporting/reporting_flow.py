# reporting_flow.py
from crewai import Crew, Process
from crewai.flow.flow import Flow, start
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from .dependencies import ReportingDeps
from .tasks import shap_explanations_task, risk_summary_task
from .agents import risk_summary_agent, shap_explainer_agent

class PatientInputs(BaseModel):
    id: int | str = -1
    name: str = ""
    age: int | None = None
    gender: str | None = None
    features: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PatientInputs":
        return cls(
            id=data.get("id", -1),
            name=data.get("name", ""),
            age=data.get("age"),
            gender=data.get("gender"),
            features=data.get("features", {}) or {},
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "features": self.features,
        }

class ReportingState(BaseModel):
    class Config: arbitrary_types_allowed = True
    inputs: Optional[PatientInputs] = None
    dependencies: Optional[ReportingDeps] = None
    result: Optional[Dict[str, Any]] = None

class ReportingFlow(Flow[ReportingState]):
    def __init__(self, inputs: PatientInputs, dependencies: ReportingDeps):
        super().__init__(ReportingState(inputs=inputs, dependencies=dependencies.ensure()))

    @start()
    def run(self):
        st = self.state
        dep = st.dependencies
        pid = st.inputs.id
        feats = st.inputs.features

        cfg = dep.configs.get(pid)
        top_k = int(cfg.get("top_k", 5))
        thresholds = {"moderate": float(cfg.get("moderate", 0.33)),
                      "high": float(cfg.get("high", 0.66))}

        prob = float(dep.prediction.predict(feats))
        shap_vals = dep.shap.topk(feats, top_k)
        label = "low" if prob < thresholds["moderate"] else ("medium" if prob < thresholds["high"] else "high")

        crew = Crew(
            agents=[shap_explainer_agent, risk_summary_agent],
            tasks=[shap_explanations_task, risk_summary_task],
            process=Process.sequential,
            verbose=True,
        )
        crew_inputs = {
            "probability": prob,
            "risk_label": label,
            "thresholds": thresholds,
            "shap": shap_vals,
            **st.inputs.to_payload(),
        }
        _ = crew.kickoff(inputs=crew_inputs)

        shap_expl = crew.tasks[0].output.pydantic 
        summary = crew.tasks[1].output.pydantic

        template_data = {
            "patient": {"id": pid, "name": st.inputs.name, "age": st.inputs.age, "gender": st.inputs.gender},
            "risk": {"prob": prob, "label": label},
            "explanations": shap_expl.model_dump(),
            "summary": summary.model_dump(),
            "shap": shap_vals,
        }
        html = dep.templater.render(template_data)

        st.result = {
            "risk": {"prob": prob, "label": label},
            "shap": shap_vals,
            "explanations": shap_expl.model_dump(),
            "summary": summary.model_dump(),
            "report": {"html": html},
        }
        return st.result

def call_reporting_flow(inputs_json: Dict[str, Any], deps: ReportingDeps) -> Dict[str, Any]:
    flow = ReportingFlow(PatientInputs.from_json(inputs_json), deps)
    return flow.run()
