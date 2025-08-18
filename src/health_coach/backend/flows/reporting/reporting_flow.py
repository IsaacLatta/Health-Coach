from crewai import Crew, Process
from crewai.flow.flow import Flow, start, listen, and_
from pydantic import BaseModel
from typing import Optional, Dict, Any
from .dependencies import ReportingDeps

from .tasks import shap_explanations_task, risk_summary_task
from .agents import risk_summary_agent, shap_explainer_agent

class PatientInputs(BaseModel):
    id: int = -1
    name: str = ""
    features: dict = {}

    def from_json(self, json):
        self.id = json.get('id')
        self.name = json.get('name')
        self.features = json.get('features', {})
        
    def to_json(self) -> Dict[str, Any]:
        ...



class ReportingState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    inputs : Optional[PatientInputs] = None
    dependencies: Optional[ReportingDeps] = None

class ReportingFlow(Flow[ReportingState]):
    def __init__(self, inputs: PatientInputs, dependencies: ReportingDeps):
        super().__init__(ReportingState(inputs=inputs, dependencies=dependencies))

    @start()
    def run(self):
        patient_id = self.state.inputs.id
        features = self.state.inputs.features

        dep = self.state.dependencies
        cfg = dep.configs.get(patient_id)
        prob = dep.prediction.predict(self.state.inputs.features)
        shap_vals = dep.shap.topk(features, cfg["top_k"])
        label = "low" if prob < cfg["moderate"] else ("medium" if prob < cfg["high"] else "high")

        
        # run the mini crew / call each agent individually
        crew = self.build_summary_crew()
        
        result = crew.kickoff(inputs=self.state.inputs.to_json())

        # parse the results

        # report = self.state.dependencies.templater.render()
        # return report

    def build_summary_crew(self) -> Crew:
        return Crew(
            agents=[shap_explainer_agent, risk_summary_agent],
            tasks=[shap_explanations_task, risk_summary_task],
            process=Process.sequential,   # run summary after explanations
            verbose=True,
        )


async def call_reporting_flow(inputs):
    # call the flow + any before/after work
    ...

    