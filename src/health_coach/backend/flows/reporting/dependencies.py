# dependencies.py
from typing import Optional
from dataclasses import dataclass
import health_coach.backend.services as svc
import health_coach.backend.stores as store

@dataclass
class ReportingDeps:
    configs: Optional[store.ConfigStore] = None
    prediction: Optional[svc.PredictionService] = None
    shap: Optional[svc.ShapService] = None
    templater: Optional[svc.TemplateService] = None

    @classmethod
    def make(cls):
        return cls()

    def with_configs(self, configs: store.ConfigStore):
        self.configs = configs
        return self

    def with_prediction(self, prediction: svc.PredictionService):
        self.prediction = prediction
        return self

    def with_shap(self, shap: svc.ShapService):
        self.shap = shap
        return self

    def with_templater(self, templater: svc.TemplateService):
        self.templater = templater
        return self

    def ensure(self) -> "ReportingDeps":
        assert self.configs is not None, "ReportingDeps missing config store"
        assert self.prediction is not None, "ReportingDeps missing prediction service"
        assert self.shap is not None, "ReportingDeps missing shap service"
        assert self.templater is not None, "ReportingDeps missing config store"
        return self