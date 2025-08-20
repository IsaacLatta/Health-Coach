# app_factory.py
from flask import Flask

import health_coach.config as cfg
from .services.prediction import MockPredictionService, SklearnPicklePredictionService
from .services.shap import MockSHAP
from .services.template import SimpleHTMLTemplate, VerboseTemplate
from .stores.config import InMemConfigs
from .flows.reporting.dependencies import ReportingDeps

from .stores.transitions import InMemTransitions
from .stores.qtable import InMemQTables
from .flows.rl.dependencies import RLDeps
from .services.context import InMemContextService
from .services.rl import QLearningRLService

from .flows.rl.tools.explorer_factories import get_factories 

def create_app() -> Flask:
    app = Flask(__name__)


    reporting_deps = (
        ReportingDeps.make()
        .with_configs(InMemConfigs())
        .with_prediction(SklearnPicklePredictionService()) 
        .with_shap(MockSHAP())                          
        .with_templater(VerboseTemplate())      
    )

    # reporting_deps = (
    #     ReportingDeps.make()
    #     .with_configs(InMemConfigs())
    #     .with_prediction(MockPredictionService())
    #     .with_shap(MockSHAP())
    #     .with_templater(SimpleHTMLTemplate())
    #     .ensure()
    # )

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
        .with_qtables(InMemQTables(cfg.Q_STATES, cfg.Q_ACTIONS))
        .with_transitions(InMemTransitions())
        .with_prediction(SklearnPicklePredictionService())
        .with_context(InMemContextService())
        .with_rl(rl_service)
        .ensure()
    )

    app.config["REPORTING_DEPS"] = reporting_deps
    app.config["RL_DEPS"] = rl_deps
    return app
