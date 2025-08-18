from abc import abstractmethod, ABC
from typing import Dict, Any

class PredictionService(ABC):
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> float:
        ...

class MockPredictionService(PredictionService):
    def predict(self, features: Dict[str, Any]) -> float:
        return 0.42
    
