import pandas as pd
from pathlib import Path
import pickle
from typing import List, Tuple, Dict, Any

from rl_data_gen.drift import simulate_drift

_DATA_SET_PATH_ = Path(__file__).resolve().parents[1]/"resources"/"data"/"Heart_Disease_Prediction.csv"
_MODEL_PATH_  = Path(__file__).resolve().parents[1]/"models"/"cv_pred_log_reg.pkl"

FEATURE_NAMES = [
    "Age","Sex","Chest pain type","BP","Cholesterol","FBS over 120",
    "EKG results","Max HR","Exercise angina","ST depression",
    "Slope of ST","Number of vessels fluro","Thallium"
]

def get_model():
    with open(_MODEL_PATH_, "rb") as f:
        return pickle.load(f)


def make_prediction(model, features: List[Any]) -> float:
    return float(model.predict_proba([features])[0][1])


def extract_anchor_pairs(csv_path: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    n = len(records)
    even_count = n if (n % 2 == 0) else (n - 1)
    if even_count < 2:
        return []

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for i in range(0, even_count, 2):
        pairs.append((records[i], records[i+1]))
    return pairs


def main():
    print(f"Extract anchor pairs from dataset: '{_DATA_SET_PATH_}'...")
    anchors = extract_anchor_pairs(_DATA_SET_PATH_)
    print(f"Found {len(anchors)} anchor pairs. Simulating drifts...\n")
    if not anchors:
        print("No anchors found, exiting.")
        return

    model = get_model()

    row0, row1 = anchors[0]
    feat0 = [float(row0[col]) for col in FEATURE_NAMES]  
    feat1 = [float(row1[col]) for col in FEATURE_NAMES]  
    p_start = make_prediction(model, feat0)
    p_end   = make_prediction(model, feat1)

    print(f"Anchor 0 features -> p_start={p_start:.4f}")
    print(f"Anchor 1 features -> p_end  ={p_end:.4f}\n")
    trajectory = simulate_drift(p_start, p_end, seed=42)

    print("First 10 steps of synthetic drift trajectory:")
    for t, step in enumerate(trajectory[:10]):
        print(f"Step {t:2d}: ", step)

    print(f"\nGenerated trajectory length: {len(trajectory)}")

if __name__ == "__main__":
    main()
