# app_factory.py
from flask import Flask
from .services.prediction import MockPredictionService  # swap later
from .services.shap import MockSHAP
from .services.template import SimpleHTMLTemplate
from .stores.config import InMemConfigs
from .flows.reporting.dependencies import ReportingDeps
from .flows.rl.dependencies import RLDeps  # your RL deps

def create_app() -> Flask:
    app = Flask(__name__)

    # Build DI for flows (mock -> real later)
    reporting_deps = (ReportingDeps.builder()
        .with_configs(InMemConfigs())
        .with_predict(MockPredictionService())
        .with_shap(MockSHAP())
        .with_templater(SimpleHTMLTemplate())
        .ensure())

    rl_deps = RLDeps.builder() \
        .with_engine(...) \
        .with_transitions_store(...) \
        .with_qtable_store(...) \
        .ensure()

    app.config["REPORTING_DEPS"] = reporting_deps
    app.config["RL_DEPS"] = rl_deps
    return app
