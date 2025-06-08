# in health_coach/tools/report.py
from crewai.tools import tool
from pathlib import Path
from typing import List, Dict
import re
from datetime import datetime

_TEMPLATE_ = Path(__file__).resolve().parents[3]/"resources"/"templates"/"report_template.html"
_TEMPLATE_HTML_ = _TEMPLATE_.read_text()
_TEMPLATE_OUTPUT_PATH_ = Path(__file__).resolve().parents[3]/"out"/"patient_report.html"

@tool
def generate_report(features: List[float], probability: float, explanation: str) -> str:
    """
    features: list of floats in the same order the model expects.
    probability: the probability returned from the heart disease predictor
    explanation: explanation of the shap values of the prediction
    returns: probability of heart disease (float between 0 and 1).
    """
    html = _TEMPLATE_HTML_

    keys = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"]
    for key, value in zip(keys, features):
        pattern = re.compile(r"\{\{\s*"+key+r"\s*\}\}")
        html = pattern.sub(str(value), html)

    pct = round(probability * 100, 2)
    html = re.sub(r"\{\{\s*probability_percent\s*\}\}", f"{pct}", html)
    html = re.sub(r"\{\{\s*probability\s*\}\}", f"{probability:.4f}", html)

    html = re.sub(r"\{\{\s*explanation\s*\}\}", explanation, html)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = re.sub(r"\{\{\s*date\s*\}\}", now, html)

    return html

@tool 
def save_report(html: str) -> bool:
    """
    Save the filled in html report template to disk
    html: the filled in html report template
    returns: true on success, false on errors
    """
    try:
        _TEMPLATE_OUTPUT_PATH_.write_text(html, encoding="utf-8")
        return True
    except Exception as e:
        return False