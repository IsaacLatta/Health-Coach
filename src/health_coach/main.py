import sys
from pprint import pprint

import health_coach.config as cfg

from health_coach.backend.flows.reporting.reporting_flow import call_reporting_flow
from health_coach.backend.flows.reporting.dependencies import ReportingDeps
from health_coach.backend.services.prediction import MockPredictionService, SklearnPicklePredictionService
from health_coach.backend.services.shap import MockSHAP
from health_coach.backend.services.template import SimpleHTMLTemplate
from health_coach.backend.stores.config import InMemConfigs, SQLiteConfigs

from health_coach.backend.flows.rl.rl_flow import call_rl_flow
from health_coach.backend.flows.rl.dependencies import RLDeps
from health_coach.backend.flows.insights.dependencies import InsightsDeps
from health_coach.backend.flows.insights.insights_flow import call_insights_flow

from health_coach.backend.services.context import InMemContextService
from health_coach.backend.services.rl import QLearningRLService

from health_coach.backend.stores.qtable import InMemQTables, SQLiteQTables
from health_coach.backend.stores.transitions import InMemTransitions, SQLiteTransitions
from health_coach.backend.flows.rl.tools.explorer_factories import get_factories


def _discretize(prob: float, bins: int = 10) -> int:
    s = int(float(prob) * bins)
    return max(0, min(bins - 1, s))


def _sample_patient_and_features():
    patient = {
        "id": "P-0001",
        "name": "Jane Doe",
        "age": 54,
        "gender": "Female",
    }
    features = {
        "systolic_bp": 138,
        "diastolic_bp": 86,
        "bmi": 29.4,
        "smoker": "no",
        "hdl": 42,
        "ldl": 131,
        "family_history": "yes",
        "activity_minutes_per_week": 60,
    }
    return patient, features


def test_reporting():
    """Build reporting deps (in-mem) and run the report flow once with a sample payload."""
    reporting_deps = (
        ReportingDeps.make()
        .with_configs(InMemConfigs())
        .with_prediction(MockPredictionService())
        .with_shap(MockSHAP())
        .with_templater(SimpleHTMLTemplate())
    )

    patient, features = _sample_patient_and_features()

    payload = {
        "id": patient["id"],
        "name": patient["name"],
        "age": patient["age"],
        "gender": patient["gender"],
        "features": features,
    }

    print("\n[reporting] running call_reporting_flow() …")
    out = call_reporting_flow(payload, reporting_deps)
    result = {
        "risk": out.get("risk"),
        "shap_len": len(out.get("shap", [])),
        "has_html": bool(out.get("report", {}).get("html")),
    }
    pprint(result)
    return out


def test_rl():
    """Build RL deps (context + explorers + q/transition stores) and run a single RL step."""
    factories = get_factories()
    explorers = [
        factories["epsilon_greedy_fn"](epsilon=getattr(cfg, "EPSILON", 0.1)),
        factories["softmax_fn"](temperature=getattr(cfg, "SOFTMAX_TEMP", 0.5)),
        factories["ucb_fn"](c=getattr(cfg, "UCB_C", 1.0)),
        factories["count_bonus_fn"](beta=getattr(cfg, "COUNT_BONUS_BETA", 0.1)),
        factories["thompson_fn"](sigma=getattr(cfg, "THOMPSON_SIGMA", 1.0)),
        factories["maxent_fn"](alpha=getattr(cfg, "MAXENT_ALPHA", 0.5)),
    ]

    def reward_fn(prev: int, cur: int) -> float:
        return 1.0 if cur < prev else (-1.0 if cur > prev else 0.0)

    rl_service = QLearningRLService(
        explorers=explorers,
        reward_fn=reward_fn,
        alpha=getattr(cfg, "Q_ALPHA", 0.4),
        gamma=getattr(cfg, "Q_GAMMA", 0.9),
        verbose=getattr(cfg, "RL_VERBOSE", True),
    )

    qtables = InMemQTables(cfg.Q_STATES, cfg.Q_ACTIONS)
    transitions = InMemTransitions()
    context = InMemContextService()

    pred = MockPredictionService()

    rl_deps = (
        RLDeps.make()
        .with_qtables(qtables)
        .with_transitions(transitions)
        .with_prediction(pred)
        .with_context(context)
        .with_rl(rl_service)
        .ensure()
    )

    patient_id = 1
    _, features = _sample_patient_and_features()
    prob = float(pred.predict(features))
    curr_state = _discretize(prob, getattr(cfg, "Q_STATES", 10))
    prev_state = min(curr_state + 1, getattr(cfg, "Q_STATES", 10) - 1)

    print("\n[rl] running call_rl_flow() …")
    out = call_rl_flow(rl_deps, patient_id, prev_state, curr_state)
    pprint(out)
    return out

def test_insights():

    deps = (
    InsightsDeps.make()
    .with_prediction(SklearnPicklePredictionService())   # :contentReference[oaicite:13]{index=13}
    .with_configs(SQLiteConfigs())                        # :contentReference[oaicite:14]{index=14}
    .with_transitions(SQLiteTransitions())                # :contentReference[oaicite:15]{index=15}
    .with_qtables(SQLiteQTables(states=cfg.Q_STATES, actions=cfg.Q_ACTIONS))  # :contentReference[oaicite:16]{index=16}
    .with_context(InMemContextService())                  # :contentReference[oaicite:17]{index=17}
    # .with_templater(SimpleHTMLTemplate())              # optional, if you want HTML here
    .ensure()
    )

    payload = {
    "id": "P-0001",
    "name": "Jane Doe",
    "age": 54,
    "gender": "Female",
    "features": {...}   # same dict you send to /api/report
    }

    patient, features = _sample_patient_and_features()
    payload = {**patient, "features": features}
    out = call_insights_flow(payload, deps)
    from pprint import pprint
    pprint(out.get("summary"))
    return out