# app_factory.py
from flask import Flask
from .services.prediction import MockPredictionService
from .services.shap import MockSHAP
from .services.template import SimpleHTMLTemplate
from .stores.transitions import InMemTransitions
from .stores.config import InMemConfigs
from .stores.qtable import InMemQTables
from .flows.reporting.dependencies import ReportingDeps
from .flows.rl.dependencies import RLDeps
from .flows.rl.rl_engine import SimpleQLearningEngine

def create_app() -> Flask:
    app = Flask(__name__)

    reporting_deps = ReportingDeps.make()  \
        .with_configs(InMemConfigs()) \
        .with_predict(MockPredictionService()) \
        .with_shap(MockSHAP()) \
        .with_templater(SimpleHTMLTemplate()) \

    rl_deps = RLDeps.make() \
        .with_engine(SimpleQLearningEngine()) \
        .with_transitions(InMemTransitions()) \
        .with_qtable(InMemQTables()) \
        .with_predict(MockPredictionService())

    app.config["REPORTING_DEPS"] = reporting_deps
    app.config["RL_DEPS"] = rl_deps
    return app
