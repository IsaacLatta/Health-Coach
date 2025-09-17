# rl.py  (drop-in)
from abc import ABC, abstractmethod
from typing import List, Any, Callable, Dict

from crewai import Crew, Process
from health_coach.backend.flows.rl.agents import explorer_agent
from health_coach.backend.flows.rl.tasks import select_explorer_task
from health_coach.backend.flows.rl.tools.explorer_selector import get_hyperparams_for_explorer

QTable = List[List[float]]
State = int
Action = int
Explorer_Index = int


class RLService(ABC):
    @abstractmethod
    def possible_actions(self) -> List[int]: ...
    @abstractmethod
    def env_reward(self, prev: State, curr: State) -> float: ...
    @abstractmethod
    def select_action(self, prev: State, cur: State, reward: float, context: Any, qtable: QTable) -> tuple[Action, Explorer_Index]: ...
    @abstractmethod
    def update(self, prev: State, action: Action, reward: float, cur: State, q: QTable, explr_idx: int = 0) -> QTable: ...

class QLearningRLService(RLService):
    def __init__(self,
                 explorers: List[Callable[[State, QTable], Action]],
                 reward_fn: Callable[[State, State], float],
                 alpha: float,
                 gamma: float,
                 verbose: bool = True):
        self._explorers = explorers
        self._reward_fn = reward_fn
        self._alpha = alpha
        self._gamma = gamma
        self._verbose = verbose

    def possible_actions(self) -> List[int]:
        return list(range(len(self._explorers)))

    def env_reward(self, prev: State, curr: State) -> float:
        return float(self._reward_fn(prev, curr))

    def select_action(self, prev: State, cur: State, reward: float, context: Any, qtable: QTable) -> Action:
        inputs: Dict[str, Any] = {
            "previous_state": prev,
            "current_state": cur,
            "pure_reward": reward,
            "context": context,
        }
        try:
            crew = Crew(
                agents=[explorer_agent],
                tasks=[select_explorer_task],
                process=Process.sequential,
                verbose=self._verbose,
            )
            raw = crew.kickoff(inputs=inputs)
            idx = self._safe_select(raw["select"], len(self._explorers))
        except Exception as e:
            idx = 0 

        try:
            action = self._explorers[idx](cur, qtable)
            return (action, idx)
        except Exception as e:
            return (0, 0)

    def update(self, prev: State, action: Action, reward: float, cur: State, q: QTable, explr_idx: int = 0) -> QTable:
        if not q or not isinstance(q[0], list):
            raise ValueError("Q-table must be a list of lists[states][actions].")

        best_next = max(q[cur]) if q[cur] else 0.0
        old_q = q[prev][action]

        alpha, gamma = get_hyperparams_for_explorer(explr_idx)
        new_q = (1 - alpha) * old_q + alpha * (reward + gamma * best_next)
        q[prev][action] = float(new_q)
        return q

    def _safe_select(self, val: int, num_strats: int, default: int = 0) -> int:
        try:
            if isinstance(val, int) and 0 <= val < num_strats:
                return val
            else:
                return default
        except Exception:
            pass
        return default
