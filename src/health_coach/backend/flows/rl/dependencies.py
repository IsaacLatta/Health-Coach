from typing import Optional
from dataclasses import dataclass

import health_coach.backend.services as svc
import health_coach.backend.stores as store

from health_coach.backend.services.context import ContextService
from health_coach.backend.services.rl import RLService

@dataclass
class RLDeps:
    qtables: Optional[store.QTableStore] = None
    transitions: Optional[store.TransitionStore] = None

    prediction: Optional[svc.PredictionService] = None
    context: Optional[ContextService] = None
    rl: Optional[RLService] = None

    @classmethod
    def make(cls) -> "RLDeps":
        return cls()

    def with_qtables(self, qtables: store.QTableStore) -> "RLDeps":
        self.qtables = qtables
        return self

    def with_transitions(self, transitions: store.TransitionStore) -> "RLDeps":
        self.transitions = transitions
        return self

    def with_prediction(self, prediction: svc.PredictionService) -> "RLDeps":
        self.prediction = prediction
        return self

    def with_context(self, context: ContextService) -> "RLDeps":
        self.context = context
        return self

    def with_rl(self, rl: RLService) -> "RLDeps":
        self.rl = rl
        return self

    def ensure(self) -> "RLDeps":
        assert self.qtables is not None, "RLDeps missing qtables store"
        assert self.context is not None, "RLDeps missing ContextService"
        assert self.rl is not None, "RLDeps missing RLService"
        return self
