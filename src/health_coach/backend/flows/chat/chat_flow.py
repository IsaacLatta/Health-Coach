# backend/flows/chat/chat_flow.py
from __future__ import annotations
import json
from statistics import pstdev
from typing import Any, Dict, Optional, List

import numpy as np
from pydantic import BaseModel, Field
from crewai import Crew, Process
from crewai.flow.flow import Flow, start

import health_coach.config as cfg
from .dependencies import ChatDeps
from .tasks import chat_task, ChatReply


def _cfg_value(row: Any, name: str, default: Any) -> Any:
    """Best-effort field accessor (row may be dict/row/dataclass)."""
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(name, default)
    try:
        return getattr(row, name)
    except Exception:
        pass
    try:
        return row[name]
    except Exception:
        return default


def _margins(prob: float, thresholds: Dict[str, float]) -> Dict[str, float]:
    return {
        "margin_to_moderate": float(prob - thresholds.get("moderate", 0.33)),
        "margin_to_high": float(thresholds.get("high", 0.66) - prob),
    }


def _recent_transitions(store, patient_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    connect = getattr(store, "_connect", None)
    if callable(connect):
        with connect() as con:
            cur = con.execute(
                "SELECT ts, prev_state, action, reward, curr_state, prob "
                "FROM transitions WHERE patient_id=? ORDER BY id DESC LIMIT ?",
                (str(patient_id), int(limit)),
            )
            for ts, ps, a, r, cs, p in cur.fetchall():
                rows.append(
                    {
                        "ts": int(ts),
                        "prev_state": int(ps),
                        "action": int(a),
                        "reward": float(r),
                        "curr_state": int(cs),
                        "prob": None if p is None else float(p),
                    }
                )
        return rows

    get_last = getattr(store, "get_last", None)
    if callable(get_last):
        try:
            items = get_last(patient_id, limit=limit) or []
            for it in items:
                rows.append(
                    {
                        "ts": int(_cfg_value(it, "ts", 0)),
                        "prev_state": int(_cfg_value(it, "prev_state", 0)),
                        "action": int(_cfg_value(it, "action", -1)),
                        "reward": float(_cfg_value(it, "reward", 0.0)),
                        "curr_state": int(_cfg_value(it, "curr_state", 0)),
                        "prob": _cfg_value(it, "prob", None),
                    }
                )
        except Exception:
            pass
    return rows


def _volatility(probs: List[float]) -> Dict[str, float | str]:
    """Classify volatility similar to the insights flow."""
    if len(probs) < 2:
        return {"volatility": "low", "volatility_value": 0.0}
    v = float(pstdev(probs))
    if v < 0.02:
        lvl = "low"
    elif v < 0.06:
        lvl = "medium"
    else:
        lvl = "high"
    return {"volatility": lvl, "volatility_value": v}


def _action_stats(transitions: List[Dict[str, Any]], window: int = 30) -> List[Dict[str, Any]]:
    acc: Dict[int, List[float]] = {}
    for t in transitions[-window:]:
        a = int(t.get("action", -1))
        if a >= 0:
            acc.setdefault(a, []).append(float(t.get("reward", 0.0)))
    out: List[Dict[str, Any]] = []
    for a, rewards in sorted(acc.items()):
        mean_r = (sum(rewards) / len(rewards)) if rewards else 0.0
        out.append({"action": a, "count": len(rewards), "mean_reward": mean_r})
    return out


def _build_context_json(
    deps: ChatDeps,
    patient_id: str | int,
    features: Dict[str, Any],
    report: Optional[Dict[str, Any]],
    question: str,
) -> str:
    cfg_row = deps.configs.get(patient_id)
    thresholds = {
        "moderate": float(_cfg_value(cfg_row, "moderate", 0.33)),
        "high": float(_cfg_value(cfg_row, "high", 0.66)),
    }
    top_k = int(_cfg_value(cfg_row, "top_k", 5))

    try:
        prob = float(deps.prediction.predict(features or {}))
    except Exception:
        prob = 0.0

    last_k = _recent_transitions(deps.transitions, str(patient_id), limit=25)
    probs = [float(t["prob"]) for t in last_k if t.get("prob") is not None]
    vol = _volatility(probs)
    a_stats = _action_stats(last_k)

    q = deps.qtables.get(patient_id)
    q_info = {}
    try:
        import numpy as _np

        qa = _np.asarray(q, dtype=float)
        if qa.size:
            q_info = {
                "shape": list(qa.shape),
                "mean_q_value": float(_np.nanmean(qa)),
                "q_gap": float(_np.nanmax(qa) - _np.nanmin(qa)),
                "entropy": float(_np.nanmean(_np.log1p(_np.abs(qa)))) if _np.isfinite(qa).all() else 0.0,
            }
    except Exception:
        q_info = {}

    ctx = {
        "patient_id": str(patient_id),
        "question": question,
        "probability": prob,
        "thresholds": thresholds,
        "margins": _margins(prob, thresholds),
        "config": {"top_k": top_k},
        "transitions": last_k,
        "volatility": vol,
        "action_stats": a_stats,
        "qtable_stats": q_info,
        "report": (report or {}),
    }
    return json.dumps(ctx, ensure_ascii=False)


class ChatInputs(BaseModel):
    patient_id: int | str
    question: str
    features: Dict[str, Any] = Field(default_factory=dict)
    report: Optional[Dict[str, Any]] = None

class ChatState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    inputs: Optional[ChatInputs] = None
    deps: Optional[ChatDeps] = None
    result: Optional[Dict[str, Any]] = None

class ChatFlow(Flow[ChatState]):
    def init(self, deps: ChatDeps, inputs: ChatInputs):
        self.state.deps = deps
        self.state.inputs = inputs

    @start()
    def run(self):
        st = self.state
        dep = st.deps
        pid = st.inputs.patient_id

        ctx_json = _build_context_json(
            deps=dep,
            patient_id=pid,
            features=st.inputs.features,
            report=st.inputs.report,
            question=st.inputs.question,
        )

        crew = Crew(
            agents=[chat_task.agent],
            tasks=[chat_task],
            process=Process.sequential,
            verbose=False,
        )
        _ = crew.kickoff(inputs={"context_json": ctx_json, "question": st.inputs.question})

        reply: ChatReply = crew.tasks[0].output.pydantic
        st.result = {"reply": reply.model_dump(), "context": json.loads(ctx_json)}
        return st.result

def call_chat_flow(
    deps: ChatDeps,
    patient_id: int | str,
    question: str,
    features: Optional[Dict[str, Any]] = None,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    flow = ChatFlow()
    flow.init(
        deps=deps,
        inputs=ChatInputs(
            patient_id=patient_id,
            question=question,
            features=features or {},
            report=report,
        ),
    )
    return flow.run()
