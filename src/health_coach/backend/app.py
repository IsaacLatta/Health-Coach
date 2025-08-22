from flask import Flask

import health_coach.config as cfg
from .services.prediction import SklearnPicklePredictionService
from .services.shap import MockSHAP, SklearnSHAP
from .services.template import VerboseTemplate
from .stores.config import SQLiteConfigs
from .flows.reporting.dependencies import ReportingDeps

from .stores.transitions import SQLiteTransitions
from .stores.qtable import SQLiteQTables
from .flows.rl.dependencies import RLDeps
from .flows.chat.dependencies import ChatDeps

from .services.context import InMemContextService
from .services.rl import QLearningRLService
from .flows.rl.tools.explorer_factories import get_factories

from .flows.insights.dependencies import InsightsDeps

def create_app() -> Flask:
    app = Flask(__name__)

    reporting_deps = (
        ReportingDeps.make()
        .with_configs(SQLiteConfigs())
        .with_prediction(SklearnPicklePredictionService())
        .with_shap(SklearnSHAP())
        .with_templater(VerboseTemplate())
        .ensure()
    )

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
        verbose=True,
    )

    rl_deps = (
        RLDeps.make()
        .with_qtables(SQLiteQTables(cfg.Q_STATES, cfg.Q_ACTIONS))
        .with_transitions(SQLiteTransitions())
        .with_prediction(SklearnPicklePredictionService())
        .with_context(InMemContextService())
        .with_rl(rl_service)
        .with_configs(SQLiteConfigs())
        .ensure()
    )

    insights_deps = (
        InsightsDeps.make()
        .with_prediction(SklearnPicklePredictionService())
        .with_configs(SQLiteConfigs())
        .with_transitions(SQLiteTransitions())
        .with_qtables(SQLiteQTables(cfg.Q_STATES, cfg.Q_ACTIONS))
        .with_context(InMemContextService()) 
        .ensure()
    )

    chat_deps = (
        ChatDeps.make()
        .with_prediction(SklearnPicklePredictionService())
        .with_shap(SklearnSHAP())
        .with_configs(SQLiteConfigs())
        .with_qtables(SQLiteQTables(cfg.Q_STATES, cfg.Q_ACTIONS))
        .with_transitions(SQLiteTransitions())
        .with_context(InMemContextService())
        .ensure()
    )

    app.config["REPORTING_DEPS"] = reporting_deps
    app.config["RL_DEPS"] = rl_deps
    app.config["INSIGHTS_DEPS"] = insights_deps
    app.config["CHAT_DEPS"] = chat_deps
    return app
