from abc import abstractmethod, ABC
from typing import Dict, Any, List

class ShapService(ABC):
    @abstractmethod
    def topk(self, features: Dict[str, Any], k: int) -> List[Dict[str, Any]]: 
        ...

class MockSHAP(ShapService):
    def topk(self, features: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        items = list(features.items())[:max(1,k)]
        out = []
        for name, val in items:
            phi = round(((hash(name) % 100)/100 - 0.5)*0.2, 3)
            out.append({"feature": name, "value": val, "phi": phi, "reason": f"{name} influenced risk by {phi:+}."})
        return out