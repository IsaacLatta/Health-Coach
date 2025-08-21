from dataclasses import dataclass
from typing import Optional

from health_coach.backend.services.prediction import PredictionService
from health_coach.backend.services.context import ContextService      
from health_coach.backend.stores.transitions import TransitionStore   
from health_coach.backend.stores.qtable import QTableStore            
from health_coach.backend.stores.config import ConfigStore            
from health_coach.backend.services.template import TemplateService

@dataclass
class InsightsDeps:
    prediction: Optional[PredictionService] = None
    context: Optional[ContextService] = None
    transitions: Optional[TransitionStore] = None
    qtables: Optional[QTableStore] = None
    configs: Optional[ConfigStore] = None
    templater: Optional[TemplateService] = None  

    @classmethod
    def make(cls):
        return cls()

    def with_prediction(self, p: PredictionService): self.prediction = p; return self
    def with_context(self, c: ContextService): self.context = c; return self
    def with_transitions(self, t: TransitionStore): self.transitions = t; return self
    def with_qtables(self, q: QTableStore): self.qtables = q; return self
    def with_configs(self, c: ConfigStore): self.configs = c; return self
    def with_templater(self, t: TemplateService): self.templater = t; return self

    def ensure(self):
        assert self.prediction is not None, "prediction required"
        assert self.transitions is not None, "transitions required"
        assert self.configs is not None, "configs required"
        return self
