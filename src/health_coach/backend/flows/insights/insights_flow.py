from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from statistics import pstdev

from crewai import Crew, Process
from crewai.flow.flow import Flow, start
from pydantic import BaseModel, Field

import health_coach.config as cfg
from .dependencies import InsightsDeps
from .tasks import insights_task, Metrics, ActionStat

# Reuse shape of inputs, same as reporing
class PatientInputs(BaseModel):
    id: int | str = -1
    name: str = ""
    age: int | None = None
    gender: str | None = None
    features: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PatientInputs":
        return cls(
            id=data.get("id", -1),
            name=data.get("name", ""),
            age=data.get("age"),
            gender=data.get("gender"),
            features=data.get("features", {}) or {},
        )

class InsightsState(BaseModel):
    class Config: arbitrary_types_allowed = True
    inputs: Optional[PatientInputs] = None
    dependencies: Optional[InsightsDeps] = None
    result: Optional[Dict[str, Any]] = None

def _margins(prob: float, thresholds: Dict[str, float]) -> Tuple[float,float]:
    return prob - thresholds["moderate"], thresholds["high"] - prob

def _config_deltas(current: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, dv in defaults.items():
        cv = current.get(k, dv)
        try:
            out[k] = float(cv) - float(dv)
        except Exception:
            pass
    return out

def _recent_transitions_sqlite(store, patient_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    connect = getattr(store, "_connect", None) 
    if connect is None:
        return rows
    with connect() as con:
        cur = con.execute(
            "SELECT ts, prev_state, action, reward, curr_state, prob "
            "FROM transitions WHERE patient_id=? ORDER BY id DESC LIMIT ?",
            (str(patient_id), int(limit))
        )
        for ts, ps, a, r, cs, p in cur.fetchall():
            rows.append({
                "ts": int(ts), "prev_state": int(ps), "action": int(a),
                "reward": float(r), "curr_state": int(cs),
                "prob": None if p is None else float(p),
            })
    return rows

def _volatility(probs: List[float]) -> Tuple[str, float]:
    if len(probs) < 2:
        return "low", 0.0
    v = float(pstdev(probs))  # population stdev; tiny horizon
    if v < 0.02:  lvl = "low"
    elif v < 0.06: lvl = "medium"
    else:          lvl = "high"
    return lvl, v

class RLInsightsFlow(Flow[InsightsState]):
    def init(self, inputs: PatientInputs, dependencies: InsightsDeps):
        self.state.inputs = inputs
        self.state.dependencies = dependencies

    @start()
    def run(self):
        st = self.state
        dep = st.dependencies
        pid = st.inputs.id
        feats = st.inputs.features

        prob = float(dep.prediction.predict(feats))
        current_cfg = dep.configs.get(pid)
        defaults = cfg.default_patient_config() if hasattr(cfg, "default_patient_config") else {"moderate":0.33,"high":0.66,"top_k":5}
        thresholds = {"moderate": float(current_cfg.get("moderate", defaults["moderate"])),
                      "high": float(current_cfg.get("high", defaults["high"]))}

        m_mod, m_high = _margins(prob, thresholds)

        recent = _recent_transitions_sqlite(dep.transitions, str(pid), limit=12)
        probs = [r["prob"] for r in recent if r.get("prob") is not None]
        vol_label, vol_val = _volatility(probs)

        by_action: Dict[int, List[float]] = {}
        for r in recent:
            a = int(r["action"])
            by_action.setdefault(a, []).append(float(r["reward"]))
        action_stats = []
        for a, rewards in sorted(by_action.items()):
            if rewards:
                action_stats.append(ActionStat(action=a, count=len(rewards), mean_reward=sum(rewards)/len(rewards)))

        deltas = _config_deltas(current_cfg, defaults)

        ctx_json = None
        if dep.context is not None and dep.qtables is not None and recent:
            prev_state = int(recent[0]["prev_state"])
            curr_state = int(recent[0]["curr_state"])
            reward = float(recent[0]["reward"])
            q = dep.qtables.get(pid)
            try:
                ctx_json = dep.context.pre_action(self.inputs.id, prev_state, curr_state, reward, q)
            except Exception:
                ctx_json = None

        metrics = Metrics(
            probability=prob,
            thresholds=thresholds,
            margin_to_moderate=m_mod,
            margin_to_high=m_high,
            volatility=vol_label,
            volatility_value=vol_val,
            config_deltas=deltas,
            action_stats=action_stats,
            notes="Context snapshot included" if ctx_json else None,
        )

        crew = Crew(
            agents=[insights_task.agent],
            tasks=[insights_task],
            process=Process.sequential,
            verbose=False,
        )
        _ = crew.kickoff(inputs={"metrics_json": metrics.model_dump_json()})

        summary = insights_task.output.pydantic

        html = None
        if dep.templater is not None:
            payload = {
                "insights": {
                    "narrative": summary.narrative,
                    "bullets": summary.bullets,
                    "margins": {"to_moderate": m_mod, "to_high": m_high},
                    "volatility": {"label": vol_label, "value": vol_val},
                    "config_deltas": deltas,
                    "actions": [a.model_dump() for a in action_stats],
                }
            }
            try:
                html = dep.templater.render(payload)
            except Exception:
                html = None

        st.result = {
            "metrics": metrics.model_dump(),
            "summary": summary.model_dump(),
            "context": ctx_json,
            "html": html,
        }
        return st.result

def call_insights_flow(inputs_json: Dict[str, Any], deps: InsightsDeps) -> Dict[str, Any]:
    flow = RLInsightsFlow()
    flow.init(PatientInputs.from_json(inputs_json), deps)
    return flow.run()
