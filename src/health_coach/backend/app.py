# app_factory.py
from flask import Flask

import health_coach.config as cfg
from .services.prediction import MockPredictionService
from .services.shap import MockSHAP
from .services.template import SimpleHTMLTemplate
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

    # ---------- Reporting DI ----------
    reporting_deps = (
        ReportingDeps.make()
        .with_configs(InMemConfigs())
        .with_predict(MockPredictionService())
        .with_shap(MockSHAP())
        .with_templater(SimpleHTMLTemplate())
    )

    # ---------- RL DI ----------

    # 1) Build explorers from config-tuned hyperparameters
    # Factories: epsilon_greedy_fn, softmax_fn, ucb_fn, count_bonus_fn, thompson_fn, maxent_fn
    factories = get_factories()
    explorers = [
        factories["epsilon_greedy_fn"](epsilon=getattr(cfg, "EPSILON", 0.1)),
        factories["softmax_fn"](temperature=getattr(cfg, "SOFTMAX_TEMP", 0.5)),
        factories["ucb_fn"](c=getattr(cfg, "UCB_C", 1.0)),
        factories["count_bonus_fn"](beta=getattr(cfg, "COUNT_BONUS_BETA", 0.1)),
        factories["thompson_fn"](sigma=getattr(cfg, "THOMPSON_SIGMA", 1.0)),
        factories["maxent_fn"](alpha=getattr(cfg, "MAXENT_ALPHA", 0.5)),
    ]
    # Each explorer has signature: (state: int, q_table: np.ndarray) -> int

    # 2) Reward and policy updater (unchanged)
    def reward_fn(prev: int, cur: int) -> float:
        return 1.0 if cur < prev else (-1.0 if cur > prev else 0.0)

    rl_service = QLearningRLService(
        explorers=explorers,
        reward_fn=reward_fn,
        alpha=getattr(cfg, "Q_ALPHA", 0.4),
        gamma=getattr(cfg, "Q_GAMMA", 0.9),
        verbose=getattr(cfg, "RL_VERBOSE", True),
    )

    rl_deps = (
        RLDeps.make()
        .with_qtables(InMemQTables())
        .with_transitions(InMemTransitions())
        .with_prediction(MockPredictionService())  # keep if your flow discretizes via prob
        .with_context(InMemContextService())
        .with_rl(rl_service)
        .ensure()
    )

    app.config["REPORTING_DEPS"] = reporting_deps
    app.config["RL_DEPS"] = rl_deps
    return app
