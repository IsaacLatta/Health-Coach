from typing import Optional
from dataclasses import dataclass

from .rl_engine import RLEngine
import health_coach.backend.services as svc
import health_coach.backend.stores as store

@dataclass
class RLDeps:
    qtable: Optional[store.QTableStore] = None
    transitions: Optional[store.TransitionStore] = None
    prediction: Optional[svc.PredictionService] = None
    engine: Optional[RLEngine] = None
    # later add the context engine
    
    @classmethod
    def builder(cls):
        return cls()
    
    def with_qtable(self, qtable: store.QTableStore):
        self.qtable = qtable
        return self
    
    def with_transitions(self, transitions: store.TransitionStore):
        self.transitions = transitions
        return self
    
    def with_predict(self, predict: svc.PredictionService):
        self.prediction = predict
        return self
    
    def with_engine(self, engine: RLEngine):
        self.engine = engine
        return self