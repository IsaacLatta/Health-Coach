from abc import ABC, abstractmethod
import json
from collections import deque, defaultdict
from typing import Dict, Optional, List, Any
import numpy as np
import health_coach.config as cfg

QTable = List[List[float]]
class ContextService(ABC):

    @abstractmethod
    def pre_action(self, 
                   prev: int, 
                   cur: int, 
                   reward: float, 
                   qtable: QTable) -> Any: 
            ...

    @abstractmethod
    def post_action(self, 
                    prev: int, 
                    cur: int, 
                    reward: float, 
                    action: int, 
                    q: QTable) -> Any: 
            ...

class InMemContextService(ContextService):
    """Single source of truth: holds rolling stats; snapshots Q locally."""
    def __init__(self, reward_buffer_size: int = 5, td_buffer_size: int = 5, action_buffer_size: int = 10):
        self.visit_counts: Dict[int, int] = defaultdict(int)
        self.action_counts: Dict[int, int] = defaultdict(int)
        self.reward_buf: deque[float] = deque(maxlen=reward_buffer_size)
        self.td_buf: deque[float] = deque(maxlen=td_buffer_size)
        self.action_buf: deque[int] = deque(maxlen=action_buffer_size)
        self.step: int = 0

        self.prev_state: Optional[int] = None
        self.cur_state: Optional[int] = None
        self.last_action: Optional[int] = None
        self.last_reward: float = 0.0
        self.q: Optional[np.ndarray] = None

    def _ctx_json(self, prev: int, cur: int, reward: float, action: Optional[int], qtable: Optional[np.ndarray]) -> str:
        ctx: Dict[str, Optional[float]] = {}
        ctx["step"] = float(self.step)
        ctx["received_reward"] = float(reward)

        visits = self.visit_counts.get(cur, 0)
        actions = self.action_counts.get(action, 0) if action is not None else 0
        ctx["visit_count"] = float(visits)
        ctx["action_count"] = float(actions)
        ctx["action_ratio"] = float(actions / visits) if visits > 0 else None

        if len(self.reward_buf) > 0:
            arr = np.array(self.reward_buf, dtype=float)
            ctx["reward_mean"] = float(arr.mean())
            ctx["reward_variance"] = float(arr.var())
            if len(arr) > 1:
                ctx["reward_trend"] = float(arr[-1] - arr[0])

        if qtable is not None:
            q_row = np.array(qtable[cur], dtype=float)
            chosen_q = float(q_row[action]) if (action is not None) else None
            ctx["chosen_q_value"] = chosen_q
            ctx["mean_q_value"] = float(q_row.mean())

            sorted_q = np.sort(q_row)
            if sorted_q.size > 1:
                ctx["q_gap"] = float(sorted_q[-1] - sorted_q[-2])
                if chosen_q is not None:
                    ctx["q_advantage"] = float(chosen_q - sorted_q[-2])

            exp_q = np.exp(q_row / cfg.SOFTMAX_TEMP)
            probs = exp_q / exp_q.sum()
            ctx["entropy"] = float(-np.sum(probs * np.log(probs + 1e-8)))

        if len(self.td_buf) > 0:
            tda = np.array(self.td_buf, dtype=float)
            ctx["td_error_mean"] = float(tda.mean())
            ctx["td_error_variance"] = float(tda.var())

        return json.dumps(ctx)

    def pre_action(self, prev: int, cur: int, reward: float, qtable: QTable) -> Any:
        """Compute a context snapshot WITHOUT mutating buffers/counts."""
        q_np = np.array(qtable, dtype=float) if qtable is not None else None
        # Show context for current state with no chosen action yet
        return self._ctx_json(prev, cur, reward, action=None, qtable=q_np)

    def post_action(self, prev: int, cur: int, reward: float, action: int, q: QTable) -> Any:
        """Commit the step to rolling stats; return updated context snapshot."""
        self.prev_state = prev
        self.cur_state = cur
        self.last_action = int(action)
        self.last_reward = float(reward)
        self.q = np.array(q, dtype=float) if q is not None else None
        self.step += 1

        self.visit_counts[cur] += 1
        self.action_counts[action] += 1
        self.reward_buf.append(float(reward))
        self.action_buf.append(int(action))

        if self.q is not None:
            best_next = float(np.max(self.q[cur]))
            td = float(reward) + cfg.GAMMA * best_next - float(self.q[prev, action])
            self.td_buf.append(td)

        return self._ctx_json(prev, cur, reward, action=action, qtable=self.q)
