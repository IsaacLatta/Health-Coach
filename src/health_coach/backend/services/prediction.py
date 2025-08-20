from abc import abstractmethod, ABC
from typing import Dict, Any, List, Optional, Mapping
from pathlib import Path
import os
import pickle

class PredictionService(ABC):
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> float:
        ...

class MockPredictionService(PredictionService):
    def predict(self, features: Dict[str, Any]) -> float:
        return 0.42

# Feature order used when the model was trained (UCI heart-like schema)
FEATURE_ORDER: List[str] = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

# Friendly aliases possibly from UI clients
ALIASES: Mapping[str, str] = {
    "RestingBP": "BP",
    "Chol": "Cholesterol",
    "FBS": "FBS over 120",
    "Restecg": "EKG results",
    "Thalach": "Max HR",
    "Exang": "Exercise angina",
    "Oldpeak": "ST depression",
    "CP": "Chest pain type",
    "Slope": "Slope of ST",
    "Ca": "Number of vessels fluro",
    "Thal": "Thallium",
}

def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _enc_binary(v: Any, true_vals=("yes", "y", "true", "1", 1)) -> int:
    if isinstance(v, str):
        return 1 if v.strip().lower() in true_vals else 0
    return 1 if v in true_vals else 0

def _enc_sex(v: Any) -> int:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("male", "m"): return 1
        if s in ("female", "f"): return 0
    return int(_coerce_float(v))

def _enc_cp(v: Any) -> int:
    # UCI: 1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic
    if isinstance(v, str):
        s = v.strip().lower()
        mapping = {
            "typical": 1, "typical angina": 1,
            "atypical": 2, "atypical angina": 2,
            "non-anginal": 3, "nonanginal": 3, "non anginal": 3,
            "asymptomatic": 4,
        }
        if s in mapping: return mapping[s]
    return int(_coerce_float(v, 1))

def _enc_thal(v: Any) -> int:
    if isinstance(v, str):
        s = v.strip().lower()
        if "normal" in s: return 3
        if "fixed" in s: return 6
        if "revers" in s: return 7
    return int(_coerce_float(v, 3))

def _enc_slope(v: Any) -> int:
    # UCI slope often 0,1,2; allow text
    if isinstance(v, str):
        s = v.strip().lower()
        if "down" in s: return 0
        if "flat" in s: return 1
        if "up" in s: return 2
    return int(_coerce_float(v, 1))

def _normalize_feature_name(name: str) -> str:
    if name in FEATURE_ORDER:
        return name
    return ALIASES.get(name, name)

class SklearnPicklePredictionService(PredictionService):
    def __init__(self, model_path: Optional[str | Path] = None,
                 feature_order: Optional[List[str]] = None):
        self.model_path = Path(model_path or os.getenv("PRED_MODEL_PATH", "models/cv_pred_log_reg.pkl"))
        self.feature_order = feature_order or FEATURE_ORDER
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)

    def _encode_row(self, features: Dict[str, Any]) -> List[float]:
        f = { _normalize_feature_name(k): v for k, v in features.items() }

        row: List[float] = []
        for name in self.feature_order:
            v = f.get(name)

            if name == "Sex":
                row.append(float(_enc_sex(v)))
            elif name == "Chest pain type":
                row.append(float(_enc_cp(v)))
            elif name == "FBS over 120":
                # Accept yes/no OR actual glucose values
                if isinstance(v, (int, float)) and v > 1:
                    row.append(float(1 if v > 120 else 0))
                else:
                    row.append(float(_enc_binary(v)))
            elif name == "Exercise angina":
                row.append(float(_enc_binary(v)))
            elif name == "Thallium":
                row.append(float(_enc_thal(v)))
            elif name == "Slope of ST":
                row.append(float(_enc_slope(v)))
            else:
                row.append(_coerce_float(v, 0.0))
        return row

    def predict(self, features: Dict[str, Any]) -> float:
        self._ensure_model()
        row = self._encode_row(features)
        proba = self._model.predict_proba([row])[0][1]
        return float(proba)
