# src/health_coach/tools/explain.py
import numpy as np
import shap
from typing import List
from crewai.tools import tool
from health_coach.tools.prediction import get_model


@tool
def generate_prediction_explanation(feature_names: List[str], features: List[float], top_n: int = 5) -> str:
    """
    feature_names: names of the input features to the prediction model.
    features: list of floats in the same order as the feature names. 
    returns: an HTML <ul> list of the top_n features by absolute SHAP value.
    """
    model = get_model()
    explainer = shap.LinearExplainer(
        model,
        masker=np.zeros((1, len(feature_names)))
    )
    raw = explainer.shap_values(np.array([features]))

    # pick the only set of shap values
    if isinstance(raw, list):
        shap_matrix = raw[0]
    else:
        shap_matrix = raw

    shap_vals = shap_matrix.flatten()

    # pair, sort, and take top_n
    pairs = list(zip(feature_names, shap_vals))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top = pairs[:top_n]

    items = "".join(
        f"<li><strong>{name}</strong>: {val:.3f}</li>"
        for name, val in top
    )
    return f"<h2>Feature Contributions</h2><ul>{items}</ul>"
