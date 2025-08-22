from dataclasses import dataclass
from typing import Optional
import health_coach.backend.services as svc
import health_coach.backend.stores as store

@dataclass
class ChatDeps:
    prediction: Optional[svc.PredictionService] = None
    shap: Optional[svc.ShapService] = None
    configs: Optional[store.ConfigStore] = None
    qtables: Optional[store.QTableStore] = None
    transitions: Optional[store.TransitionStore] = None
    context: Optional[svc.ContextService] = None
    
    @classmethod
    def make(cls): return cls()

    def with_prediction(self, p): self.prediction = p; return self
    def with_shap(self, s): self.shap = s; return self
    def with_configs(self, c): self.configs = c; return self
    def with_qtables(self, q): self.qtables = q; return self
    def with_transitions(self, t): self.transitions = t; return self
    def with_context(self, c): self.context = c; return self

    def ensure(self):
        assert self.configs and self.qtables and self.transitions
        assert self.prediction and self.shap and self.context
        return self
