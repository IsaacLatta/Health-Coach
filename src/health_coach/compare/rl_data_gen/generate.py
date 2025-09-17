import pandas as pd
import pickle
import random
from pathlib import Path
from typing import Any, List, Tuple, Dict

from health_coach.compare.rl_data_gen.drift import simulate_drift, simulate_drift_probs

BASE_DIR = Path(__file__).resolve().parents[4]
_DATA_SET_PATH = BASE_DIR/"data"/"kaggle_heart_disease.csv"
_MODEL_PATH = BASE_DIR/"models"/"predictor.pkl"

FEATURE_NAMES = [
    "Age","Sex","Chest pain type","BP","Cholesterol","FBS over 120",
    "EKG results","Max HR","Exercise angina","ST depression",
    "Slope of ST","Number of vessels fluro","Thallium"
]

def extract_anchor_pairs(
    csv_path: Path = _DATA_SET_PATH,
    label_column: str = "Heart Disease"
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    df = pd.read_csv(csv_path)
    df[label_column] = df[label_column].map({"Absence": 0, "Presence": 1})
    records = df.to_dict(orient="records")
    negs = [r for r in records if r[label_column] == 0]
    poss = [r for r in records if r[label_column] == 1]
    random.shuffle(negs)
    random.shuffle(poss)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    # All possible absence->presence trend anchors
    for n in negs:
        for p in poss:
            pairs.append((n, p))
    # All possible presence->absence trend anchors
    for p in poss:
        for n in negs:
            pairs.append((p, n))
    return pairs

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
        traj = simulate_drift_probs(p0, p1, seed=seed, mode=mode)
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

def get_episodes(
    num_trend: int,
    num_random: int,
    start_seed: int = 1000
) -> List[List[Dict[str, Any]]]:
    """
    Select num_trend anchor pairs for trend mode and num_random pairs for random_walk mode,
    generate the corresponding episodes, and return them in the same list-of-lists format.
    """
    model = load_model()
    anchors = extract_anchor_pairs()
    total = len(anchors)
    if num_trend + num_random > total:
        raise ValueError(f"Requested {num_trend + num_random} episodes, "
                         f"but only {total} anchors available.")
    random.shuffle(anchors)
    trend_anchors = anchors[:num_trend]
    rand_anchors = anchors[num_trend:num_trend + num_random]
    episodes = []
    episodes += generate_episodes(trend_anchors, model, mode="trend", start_seed=start_seed)
    rand_start = start_seed + len(trend_anchors)
    episodes += generate_episodes(rand_anchors, model, mode="random_walk", start_seed=rand_start)
    return episodes
