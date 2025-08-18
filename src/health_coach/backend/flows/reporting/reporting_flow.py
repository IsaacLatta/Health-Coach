from crewai.flow.flow import Flow, start, listen, and_
from crewai.flow.persistence import persist
from pydantic import BaseModel
from typing import Optional
from .dependencies import ReportingDeps

class PatientInputs(BaseModel):
    id: int = -1
    name: str = ""
    features: dict = {}

    def from_json(self, json):
        self.id = json.get('id')
        self.name = json.get('name')
        self.features = json.get('features', {})

class ReportingState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    inputs : Optional[PatientInputs] = None
    dependencies: Optional[ReportingDeps] = None

class ReportingFlow(Flow[ReportingState]):
    def __init__(self, inputs: PatientInputs, dependencies: ReportingDeps):
        self.state.inputs = inputs
        self.state.dependencies = dependencies

    @start()
    def run(self):
        patient_id = self.state.inputs.id
        features = self.state.inputs.features

        dep = self.state.dependencies
        cfg = dep.configs.get(self.state.inputs.id)
        prob = dep.prediction.predict(self.state.inputs.features)
        shap_vals = dep.shap.topk(features, cfg["top_k"])
        label = "low" if prob < cfg["moderate"] else ("medium" if prob < cfg["high"] else "high")

        # run the mini crew / call each agent individually

        report = self.state.dependencies.templater.render()
        return report


async def call_reporting_flow(inputs):
    # call the flow + any before/after work
    ...

    