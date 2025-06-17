import pandas as pd
from pathlib import Path
import pickle
import random
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

    random.seed(42)
    random.shuffle(anchors)
    split_index = int(len(anchors) * 0.1)
    random_walk_anchors = anchors[:split_index]
    trend_anchors = anchors[split_index:]

    print(f"Generating {len(trend_anchors)} trend trajectories and {len(random_walk_anchors)} random walk trajectories.\n")

    row0, row1 = trend_anchors[0]
    feat0 = [float(row0[col]) for col in FEATURE_NAMES]
    feat1 = [float(row1[col]) for col in FEATURE_NAMES]
    p_start = make_prediction(model, feat0)
    p_end = make_prediction(model, feat1)

    print(f"Anchor 0 features -> p_start={p_start:.4f}")
    print(f"Anchor 1 features -> p_end  ={p_end:.4f}\n")

    trajectory = simulate_drift(p_start, p_end, seed=42, mode="trend")
    print("First 10 steps of synthetic trend trajectory:")
    for t, step in enumerate(trajectory[:10]):
        print(f"Step {t:2d}: ", step)
    print(f"\nGenerated trajectory length: {len(trajectory)}")

    row0, row1 = random_walk_anchors[0]
    feat0 = [float(row0[col]) for col in FEATURE_NAMES]
    feat1 = [float(row1[col]) for col in FEATURE_NAMES]
    p_mid0 = make_prediction(model, feat0)
    p_mid1 = make_prediction(model, feat1)
    p_base = (p_mid0 + p_mid1) / 2.0

    neutral_trajectory = simulate_drift(p_mid0, p_mid1, seed=99, mode="trend")
    print("\nFirst 10 steps of synthetic random walk trajectory:")
    for t, step in enumerate(neutral_trajectory[:10]):
        print(f"Step {t:2d}: ", step)

if __name__ == "__main__":
    main()