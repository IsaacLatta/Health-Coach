import json
from collections import deque, defaultdict
from typing import Dict, Optional

import numpy as np
import health_coach.config as cfg


class ContextEngine:
    """
    Aggregates live training metrics to provide rich context for the shaper agent.
    """
    def __init__(self,
                 reward_buffer_size: int = 5,
                 td_buffer_size: int = 5,
                 action_buffer_size: int = 10):
        # Counters and buffers
        self.visit_counts: Dict[int, int] = defaultdict(int)
        self.action_counts: Dict[int, int] = defaultdict(int)
        self.reward_buffer: deque[float] = deque(maxlen=reward_buffer_size)
        self.td_error_buffer: deque[float] = deque(maxlen=td_buffer_size)
        self.action_buffer: deque[int] = deque(maxlen=action_buffer_size)
        self.step: int = 0

        # Last transition & Q-table
        self.prev_state: Optional[int] = None
        self.current_state: Optional[int] = None
        self.last_action: Optional[int] = None
        self.reward: float = 0.0
        self.q_table: Optional[np.ndarray] = None

    def update(self,
               prev_state: int,
               current_state: int,
               reward: float,
               action: int,
               q_table: np.ndarray):
        """
        Called each step by the engine with transition data and Q-table.
        Updates counters and buffers.
        """
        # Update basics
        self.prev_state = prev_state
        self.current_state = current_state
        self.last_action = action
        self.reward = reward
        self.q_table = q_table
        self.step += 1
        # Visit & action counts
        self.visit_counts[current_state] += 1
        self.action_counts[action] += 1
        # Buffers
        self.reward_buffer.append(reward)
        self.action_buffer.append(action)
        # TD error
        if (
            self.prev_state is not None
            and self.current_state is not None
            and self.q_table is not None
            and 0 <= self.prev_state < q_table.shape[0]
            and 0 <= self.current_state < q_table.shape[0]
        ):
            best_next = float(np.max(q_table[self.current_state]))
            td = reward + cfg.GAMMA * best_next - float(q_table[self.prev_state, self.last_action])
            self.td_error_buffer.append(td)

    def generate_context(self) -> str:
        """
        Returns a JSON string with a rich set of live context signals:
        - step index
        - received_reward, reward_mean, reward_variance, reward_trend
        - visit_count, action_count, action_ratio
        - chosen_q_value, q_gap, advantage
        - policy entropy
        - TD error mean and variance
        """
        ctx: Dict[str, Optional[float]] = {}
        # Basic
        ctx['step'] = float(self.step)
        ctx['received_reward'] = float(self.reward)
        # Counts
        visits = self.visit_counts.get(self.current_state, 0)
        actions = self.action_counts.get(self.last_action, 0)
        ctx['visit_count'] = float(visits)
        ctx['action_count'] = float(actions)
        ctx['action_ratio'] = float(actions / visits) if visits > 0 else None
        # Reward stats
        if self.reward_buffer:
            arr = np.array(self.reward_buffer)
            ctx['reward_mean'] = float(arr.mean())
            ctx['reward_variance'] = float(arr.var())
            if len(arr) > 1:
                ctx['reward_trend'] = float(arr[-1] - arr[0])
        # Q-value stats
        if self.q_table is not None and self.current_state is not None:
            q_vals = np.array(self.q_table[self.current_state])
            chosen_q = float(q_vals[self.last_action]) if self.last_action is not None else None
            ctx['chosen_q_value'] = chosen_q
            ctx['mean_q_value'] = float(q_vals.mean())
            # Q-gap and advantage
            sorted_q = np.sort(q_vals)
            if sorted_q.size > 1:
                gap = float(sorted_q[-1] - sorted_q[-2])
                ctx['q_gap'] = gap
                if self.last_action is not None:
                    ctx['q_advantage'] = chosen_q - float(sorted_q[-2])
            # Entropy
            exp_q = np.exp(q_vals / cfg.SOFTMAX_TEMP)
            probs = exp_q / exp_q.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            ctx['entropy'] = float(entropy)
        # TD-error stats
        if self.td_error_buffer:
            arr_td = np.array(self.td_error_buffer)
            ctx['td_error_mean'] = float(arr_td.mean())
            ctx['td_error_variance'] = float(arr_td.var())
        return json.dumps(ctx)
