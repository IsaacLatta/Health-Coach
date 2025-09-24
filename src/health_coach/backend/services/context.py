from abc import ABC, abstractmethod
from typing import Any, List, Dict
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
from typing import Dict, Deque, Optional, Any, List
import json
import health_coach.config as cfg

from health_coach.backend.flows.rl.tools.explorer_selector import get_hyperparams_for_explorer

QTable = List[List[float]]

class ContextService(ABC):
    @abstractmethod
    def pre_action(
        self, patient_id: str, prev: int, cur: int, reward: float, qtable: QTable
    ) -> Any: ...

    @abstractmethod
    def post_action(
        self, patient_id: str, prev: int, cur: int, reward: float, action: int, q: QTable
    ) -> Any: ...

@dataclass
class _PtCtx:
    step: int
    visit_counts: Dict[int, int]
    action_counts: Dict[int, int]
    reward_buf: Deque[float]
    td_buf: Deque[float]
    action_buf: Deque[int]
    prev_state: Optional[int] = None
    cur_state: Optional[int] = None
    last_action: Optional[int] = None
    last_reward: float = 0.0

class InMemContextService(ContextService):
    def __init__(self, reward_buf=5, td_buf=5, action_buf=10):
        self._by_patient: Dict[str, _PtCtx] = {}
        self._reward_n, self._td_n, self._act_n = reward_buf, td_buf, action_buf

    def _get(self, pid: str) -> _PtCtx:
        if pid not in self._by_patient:
            self._by_patient[pid] = _PtCtx(
                step=0,
                visit_counts=defaultdict(int),
                action_counts=defaultdict(int),
                reward_buf=deque(maxlen=self._reward_n),
                td_buf=deque(maxlen=self._td_n),
                action_buf=deque(maxlen=self._act_n),
            )
        return self._by_patient[pid]

    def _snapshot(self, ctx: _PtCtx, cur: int, reward: float, action: Optional[int], q_np: Optional[np.ndarray]) -> str:
        out: Dict[str, Any] = {
            "step": float(ctx.step),
            "received_reward": float(reward),
            "visit_count": float(ctx.visit_counts.get(cur, 0)),
            "action_count": float(ctx.action_counts.get(action, 0)) if action is not None else 0.0,
        }
        v = out["visit_count"]
        a = out["action_count"]
        out["action_ratio"] = float(a / v) if v > 0 else None

        if len(ctx.reward_buf):
            arr = np.array(ctx.reward_buf, dtype=float)
            out["reward_mean"] = float(arr.mean())
            out["reward_variance"] = float(arr.var())
            if len(arr) > 1:
                out["reward_trend"] = float(arr[-1] - arr[0])

        if q_np is not None:
            row = q_np[cur].astype(float)
            chosen_q = float(row[action]) if action is not None else None
            out["chosen_q_value"] = chosen_q
            out["mean_q_value"] = float(row.mean())
            if row.size > 1:
                sort = np.sort(row)
                out["q_gap"] = float(sort[-1] - sort[-2])
                if chosen_q is not None:
                    out["q_advantage"] = float(chosen_q - sort[-2])
            exp_q = np.exp(row / cfg.SOFTMAX_TEMP)
            probs = exp_q / exp_q.sum()
            out["entropy"] = float(-np.sum(probs * np.log(probs + 1e-8)))

        if len(ctx.td_buf):
            tda = np.array(ctx.td_buf, dtype=float)
            out["td_error_mean"] = float(tda.mean())
            out["td_error_variance"] = float(tda.var())

        return json.dumps(out)

    def pre_action(self, patient_id: str, prev: int, cur: int, reward: float, qtable: QTable) -> Any:
        ctx = self._get(patient_id)
        q_np = np.array(qtable, dtype=float) if qtable is not None else None
        return self._snapshot(ctx, cur=cur, reward=reward, action=None, q_np=q_np)

    def post_action(self, explr_idx: int, patient_id: str, prev: int, cur: int, reward: float, action: int, q: QTable) -> Any:
        ctx = self._get(patient_id)
        ctx.prev_state, ctx.cur_state = prev, cur
        ctx.last_action, ctx.last_reward = int(action), float(reward)
        ctx.step += 1
        ctx.visit_counts[cur] += 1
        ctx.action_counts[action] += 1
        ctx.reward_buf.append(float(reward))
        ctx.action_buf.append(int(action))
        q_np = np.array(q, dtype=float) if q is not None else None
        if q_np is not None:
            best_next = float(np.max(q_np[cur]))
            _, gamma = get_hyperparams_for_explorer(explr_idx)
            td = float(reward) + gamma * best_next - float(q_np[prev, action])
            ctx.td_buf.append(td)
        return self._snapshot(ctx, cur=cur, reward=reward, action=action, q_np=q_np)
