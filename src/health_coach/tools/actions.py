from typing import Union

def increase_moderate_threshold(old_config: dict, step: float = 0.05) -> dict:
    """
    Increase the 'moderate' threshold in the configuration.

    Args:
        old_config (dict): A config dict with a 'thresholds' key containing 'moderate'.
        step (float): Amount to add to the current moderate threshold (default 0.05).

    Returns:
        dict: The updated config dict. If the new threshold would exceed 0.99, returns the original config.
    """
    new_val = old_config["thresholds"]["moderate"] + step
    if new_val > 0.99:
        return old_config
    old_config["thresholds"]["moderate"] = new_val
    return old_config

def decrease_moderate_threshold(old_config: dict, step: float = 0.05) -> dict:
    """
    Decrease the 'moderate' threshold in the configuration.

    Args:
        old_config (dict): A config dict with a 'thresholds' key containing 'moderate'.
        step (float): Amount to subtract from the current moderate threshold (default 0.05).

    Returns:
        dict: The updated config dict. If the new threshold would fall below 0.0, returns the original config.
    """
    new_val = old_config["thresholds"]["moderate"] - step
    if new_val < 0.0:
        return old_config
    old_config["thresholds"]["moderate"] = new_val
    return old_config

def increase_high_threshold(old_config: dict, step: float = 0.05) -> dict:
    """
    Increase the 'high' threshold in the configuration.

    Args:
        old_config (dict): A config dict with a 'thresholds' key containing 'high'.
        step (float): Amount to add to the current high threshold (default 0.05).

    Returns:
        dict: The updated config dict. If the new threshold would exceed 0.99, returns the original config.
    """
    new_val = old_config["thresholds"]["high"] + step
    if new_val > 0.99:
        return old_config
    old_config["thresholds"]["high"] = new_val
    return old_config

def decrease_high_threshold(old_config: dict, step: float = 0.05) -> dict:
    """
    Decrease the 'high' threshold in the configuration.

    Args:
        old_config (dict): A config dict with a 'thresholds' key containing 'high'.
        step (float): Amount to subtract from the current high threshold (default 0.05).

    Returns:
        dict: The updated config dict. If the new threshold would fall below 0.0, returns the original config.
    """
    new_val = old_config["thresholds"]["high"] - step
    if new_val < 0.0:
        return old_config
    old_config["thresholds"]["high"] = new_val
    return old_config

def increase_top_k(old_config: dict, step: int = 1) -> dict:
    """
    Increase the 'top_k' parameter in the configuration.

    Args:
        old_config (dict): A config dict that may contain 'top_k' (defaults to 1 if missing).
        step (int): Amount to add to 'top_k' (default 1).

    Returns:
        dict: The updated config dict with incremented 'top_k'.
    """
    new_k = old_config.get("top_k", 1) + step
    old_config["top_k"] = new_k
    return old_config

def decrease_top_k(old_config: dict, step: int = 1) -> dict:
    """
    Decrease the 'top_k' parameter in the configuration.

    Args:
        old_config (dict): A config dict that may contain 'top_k' (defaults to 1 if missing).
        step (int): Amount to subtract from 'top_k' (default 1).

    Returns:
        dict: The updated config dict. If the new 'top_k' would fall below 1, returns the original config.
    """
    new_k = old_config.get("top_k", 1) - step
    if new_k < 1:
        return old_config
    old_config["top_k"] = new_k
    return old_config

def nop(old_config: dict) -> dict:
    """
    No-op tool: returns the configuration unmodified.

    Args:
        old_config (dict): Any configuration dict.

    Returns:
        dict: The same config dict, unchanged.
    """
    return old_config

ACTIONS = [
    "INCREASE_MODERATE_THRESHOLD",
    "DECREASE_MODERATE_THRESHOLD",
    "INCREASE_HIGH_THRESHOLD",
    "DECREASE_HIGH_THRESHOLD",
    "INCREASE_TOP_K",
    "DECREASE_TOP_K",
    "NOP",
]

ACTION_HANDLERS: dict = {
    ACTIONS[0]: increase_moderate_threshold,
    ACTIONS[1]: decrease_moderate_threshold,
    ACTIONS[2]: increase_high_threshold,
    ACTIONS[3]: decrease_high_threshold,
    ACTIONS[4]: increase_top_k,
    ACTIONS[5]: decrease_top_k,
    ACTIONS[6]: nop,
}

def retrieve_action_index(action: Union[int, str]) -> int:
    if isinstance(action, int):
        return action
    try:
        return ACTIONS.index(action)
    except (ValueError, AttributeError):
        return -1