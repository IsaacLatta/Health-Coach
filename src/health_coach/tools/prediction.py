# tools/custom_tool.py
from pathlib import Path
import pickle
from typing import List
import sklearn

from crewai.tools import tool

_MODEL_PATH_ = Path(__file__).resolve().parents[3]/"models"/"cv_pred_log_reg.pkl"
_MODEL_ = None

def get_model():
    global _MODEL_
    if _MODEL_ is None:
        with open(_MODEL_PATH_, "rb") as f:
            _MODEL_ = pickle.load(f)
    return _MODEL_

@tool
def predict_heart_disease(features: List[float]) -> float:
    """
    features: list of floats in the same order the model expects.
    returns: probability of heart disease (float between 0 and 1).
    """
    model = get_model()
    # scikit-learn’s predict_proba returns [[P(no), P(yes)]]
    return float(model.predict_proba([features])[0][1])
