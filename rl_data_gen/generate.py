import pandas as pd
import pickle
import random
from pathlib import Path
from typing import Any, List, Tuple, Dict

from rl_data_gen.drift import simulate_drift

BASE_DIR = Path(__file__).resolve().parents[1]
_DATA_SET_PATH = BASE_DIR/"data"/"Heart_Disease_Prediction.csv"
_MODEL_PATH = BASE_DIR/"models"/"cv_pred_log_reg.pkl"

FEATURE_NAMES = [
    "Age","Sex","Chest pain type","BP","Cholesterol","FBS over 120",
    "EKG results","Max HR","Exercise angina","ST depression",
    "Slope of ST","Number of vessels fluro","Thallium"
]

def extract_anchor_pairs(csv_path: Path = _DATA_SET_PATH) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    count = len(records)
    even = count if count % 2 == 0 else count - 1
    return [(records[i], records[i+1]) for i in range(0, even, 2)]

def split_anchors(
    anchors: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    random_frac: float = 0.1,
    seed: int = 42
) -> Tuple[List, List]:
    random.seed(seed)
    shuffled = anchors.copy()
    random.shuffle(shuffled)
    n_rand = max(1, int(len(shuffled) * random_frac))
    return shuffled[:n_rand], shuffled[n_rand:]

def load_model(model_path: Path = _MODEL_PATH) -> Any:
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_proba(model: Any, row: Dict[str, Any]) -> float:
    features = [float(row[col]) for col in FEATURE_NAMES]
    return float(model.predict_proba([features])[0][1])

def classify_trend(
    trend_anchors: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    model: Any
) -> Tuple[int, int]:
    down = up = 0
    for r0, r1 in trend_anchors:
        p0 = predict_proba(model, r0)
        p1 = predict_proba(model, r1)
        if p1 < p0:
            down += 1
        else:
            up += 1
    return down, up

def prepare_output_dirs(base: Path = BASE_DIR/"data"/"synthetic") -> Tuple[Path, Path]:
    train = base / "train"
    val   = base / "val"
    train.mkdir(parents=True, exist_ok=True)
    val.mkdir(parents=True, exist_ok=True)
    return train, val

def generate_episodes(
    anchors: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    model: Any,
    mode: str,
    start_seed: int = 1000
) -> List[List[Dict[str, Any]]]:
    eps = []
    seed = start_seed
    for r0, r1 in anchors:
        p0 = predict_proba(model, r0)
        p1 = predict_proba(model, r1)
        traj = simulate_drift(p0, p1, seed=seed, mode=mode)
        eps.append(traj)
        seed += 1
    return eps

def write_episodes(
    episodes: List[List[Dict[str, Any]]],
    train_dir: Path,
    val_dir: Path,
    train_frac: float = 0.8
):
    random.shuffle(episodes)
    total = len(episodes)
    cut   = int(total * train_frac)
    for idx, traj in enumerate(episodes):
        df = pd.DataFrame(traj)
        dest = train_dir if idx < cut else val_dir
        dest.joinpath(f"episode_{idx:03d}.csv").write_text(df.to_csv(index=False))

