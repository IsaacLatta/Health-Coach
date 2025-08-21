from abc import abstractmethod, ABC
from typing import Dict, Any, List

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import shap

from .prediction import SklearnPicklePredictionService

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

class SklearnSHAP(ShapService):
    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        feature_order: Optional[List[str]] = None,
    ):
        self._pred = SklearnPicklePredictionService(
            model_path=model_path, feature_order=feature_order
        )
        self._pred._ensure_model()           
        self._model = self._pred._model
        self._feature_order = self._pred.feature_order

        self._explainer = None 

    def _ensure_explainer(self):
        if self._explainer is not None:
            return
        n = len(self._feature_order)
        background = np.zeros((1, n), dtype=float)

        try:
            self._explainer = shap.LinearExplainer(self._model, background)
        except Exception:
            self._explainer = shap.explainers.Linear(self._model, background)

    def _row(self, features: Dict[str, Any]) -> List[float]:
        return self._pred._encode_row(features)  # noqa: SLF001

    def _direction(self, phi: float) -> str:
        return "increases" if phi > 0 else ("decreases" if phi < 0 else "mixed")

    def _shap_values_for_row(self, row: np.ndarray) -> np.ndarray:
        if hasattr(self._explainer, "shap_values"):
            raw = self._explainer.shap_values(row)
            if isinstance(raw, list):
                raw = raw[0]
            vals = np.asarray(raw, dtype=float)
            if vals.ndim == 2:  # (1, n_features)
                vals = vals[0]
            return vals
        else:
            expl = self._explainer(row) 
            vals = np.asarray(expl.values, dtype=float)
            if vals.ndim == 2:  # (1, n_features)
                vals = vals[0]
            elif vals.ndim == 3:  # (1, n_features, 1)
                vals = vals[0, :, 0]
            return vals

    def topk(self, features: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        self._ensure_explainer()
        row = np.asarray([self._row(features)], dtype=float)  # (1, n_features)

        vals = self._shap_values_for_row(row)  # (n_features,)
        k = max(1, int(k))
        order = np.argsort(np.abs(vals))[::-1][:k]

        items: List[Dict[str, Any]] = []
        for i in order:
            name = self._feature_order[i]
            phi = float(vals[i])
            items.append({
                "feature": name,
                "value": features.get(name),
                "phi": phi,
                
                # These extra fields are harmless (agent may overwrite/augment later)
                "direction": self._direction(phi),
                "magnitude": abs(phi),
                "reason": f"{name}={features.get(name)} {self._direction(phi)} estimated risk (ϕ={phi:+.4f}).",
            })
        return items

