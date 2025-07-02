import random
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Union
from crewai.tools import tool, BaseTool

from health_coach.tools.actions import retrieve_action_index

# for now just use the synthetic episodes
EPISODE_DIR: Path = Path(__file__).resolve().parents[4]/"data"/"synthetic"/"train"

# EPISODE_ID: int = random.randint(1, 2)
EPISODE_ID: int = 0

_Q_TABLE_PATH = Path(__file__).resolve().parents[4] / "models" / "q_table.npy"

def _get_episode_path() -> Path:
    filename = f"episode_{EPISODE_ID:03d}.csv"
    return EPISODE_DIR / filename

def _load_episode(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@tool
def get_q_table() -> np.ndarray:
    """
    Return the Raw Q Table for viewing
    """
    return np.load(_Q_TABLE_PATH, allow_pickle=False)

@tool
def get_visit_count(state: int) -> int:
    """
    Return the count of how many times `state` appears in the episode.
    """
    df = _load_episode(_get_episode_path())
    return int((df["state"] == state).sum())

@tool
def get_transition_count(prev_state: int, action: Union[int, str], next_state: int) -> int:
    """
    Return the count of transitions (state, action, next_state) in the episode. 
    If this data is unavailable the tool returns -1.
    """

    action_index = retrieve_action_index(action)
    if action_index < 0:
        return -1

    df = _load_episode(_get_episode_path())
    mask = (
        (df["state"] == prev_state)
        & (df["action"] == action)
        & (df["next_state"] == next_state)
    )
    return int(mask.sum())

@tool
def get_moving_average(window: int) -> list:
    """
    Compute a rolling mean of the 'reward' column over the last `window` steps.
    Returns a list of floats (NaN for the first window-1 entries).
    """
    df = _load_episode(_get_episode_path())
    return df["reward"].rolling(window).mean().tolist()

def _get_episode_length(path: Path) -> int:
    df = _load_episode(path)
    return int(len(df))

@tool
def get_episode_length() -> int:
    """
    Return the total number of transitions.
    """
    return _get_episode_length(_get_episode_path())

def get_all_tools() -> List[BaseTool]:
    return [get_episode_length, get_moving_average, get_transition_count, get_visit_count, get_q_table]
