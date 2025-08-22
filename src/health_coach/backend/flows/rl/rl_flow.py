from typing import Any, Optional, Union, Dict
import health_coach.config as cfg
from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen, and_
from .dependencies import RLDeps
from health_coach.backend.stores.transitions import Transition

def _discretize(prob: float, bins: int) -> int:
    s = int(float(prob) * bins)
    return max(0, min(bins - 1, s))

class RLState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    deps: Optional[RLDeps] = None
    patient_id: Optional[Union[int, str]] = None
    prev_state: Optional[int] = None
    curr_state: Optional[int] = None

    env_reward: float = 0.0
    context_json: Any = None
    action: int = 0
    shaped_reward: float = 0.0
    result: Optional[Dict[str, Any]] = None


class RLFlow(Flow[RLState]):

    def init(self, deps: RLDeps, patient_id, prev_state, curr_state):
        self.state.deps = deps
        self.state.patient_id = patient_id
        self.state.prev_state = prev_state
        self.state.curr_state = curr_state

    @start()
    def compute_reward(self):
        s = self.state
        s.env_reward = float(s.deps.rl.env_reward(s.prev_state, s.curr_state))
        return s.env_reward

    @listen(compute_reward)
    def make_context(self):
        s = self.state
        q = s.deps.qtables.get(s.patient_id)
        s.context_json = s.deps.context.pre_action(s.patient_id, s.prev_state, s.curr_state, s.env_reward, q)
        return s.context_json

    @listen(make_context)
    def choose_action(self):
        s = self.state
        q = s.deps.qtables.get(s.patient_id)
        s.action = int(s.deps.rl.select_action(s.prev_state, s.curr_state, s.env_reward, s.context_json, q))
        return s.action

    @listen(make_context)
    def shape_reward(self):
        self.state.shaped_reward = float(self.state.env_reward)
        return self.state.shaped_reward

    @listen(and_(choose_action, shape_reward))
    def update_q_and_context(self):
        s = self.state
        q = s.deps.qtables.get(s.patient_id)
        qn = s.deps.rl.update(s.prev_state, s.action, s.shaped_reward, s.curr_state, q)
        s.deps.qtables.save(s.patient_id, qn)

        if s.deps.transitions:
            s.deps.transitions.update(
                s.patient_id,
                Transition(
                    prev_state=s.prev_state,
                    curr_state=s.curr_state,
                    reward=int(s.shaped_reward),
                    action=int(s.action),
                )
            )

        s.deps.context.post_action(s.patient_id, s.prev_state, s.curr_state, s.shaped_reward, s.action, qn)
        s.result = {"action": s.action, "reward": s.shaped_reward, "next_state": s.curr_state}
        return s.result


def call_rl_flow(deps: RLDeps, patient_id, prev_state: int, curr_state: int) -> dict:
    flow = RLFlow()
    flow.init(deps=deps, patient_id=patient_id, prev_state=prev_state, curr_state=curr_state)
    return flow.kickoff()

def call_rl_flow_from_payload(deps: RLDeps, patient: Dict[str, Any], features: Dict[str, Any]) -> dict:
    """
    1) Predict probability from features (backend-owned).
    2) Discretize -> curr_state.
    3) Pull last transition -> prev_state (else prev=curr).
    4) Run RL flow (select action, update Q, write transition).
    5) Apply action to patient's reporting config and persist.
    """
    patient_id = patient.get("id") or patient.get("patient_id") or "-"

    prob = float(deps.prediction.predict(features))
    bins = int(getattr(cfg, "Q_STATES", 10))
    curr_state = _discretize(prob, bins)

    prev_state = curr_state
    if deps.transitions:
        try:
            last = deps.transitions.get_last(patient_id)
            if last:
                prev_state = int(getattr(last, "curr_state", curr_state))
        except Exception as e:
            print(f"Exception: {e}")
            pass  # fallback to current state

    result = call_rl_flow(deps, patient_id, prev_state, curr_state)

    try:
        raw_idx = int(result.get("action", 0))
        safe_idx = int(raw_idx) % int(getattr(cfg, "Q_ACTIONS", 6))

        if hasattr(deps, "configs") and deps.configs is not None:
            print("\nWritiing configs\n")
            current_cfg = deps.configs.get(patient_id)  # returns defaults if missing
            new_cfg = cfg.apply_config_action(current_cfg, safe_idx)
            deps.configs.save(patient_id, new_cfg)
        else:
            print("\nConfigs is none\n")
    except Exception as e:
        print(f"Exception: {e}")
        pass

    return result