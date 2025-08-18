from typing import Optional
from dataclasses import dataclass

import backend.services as svc
import backend.stores as store

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
    
    def with_predict(self, predict: svc.PredictionService):
        self.predict = predict
        return self
    
    def with_shap(self, shap: svc.ShapService):
        self.shap = shap
        return self
    
    def with_templater(self, templater: svc.TemplateService):
        self.templater = templater
        return self