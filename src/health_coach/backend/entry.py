from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, current_app
from .app import create_app
from .flows.reporting.reporting_flow import call_reporting_flow
from .flows.rl.rl_flow import call_rl_flow_from_payload
from .flows.insights.insights_flow import call_insights_flow  # NEW
from .flows.chat.chat_flow import call_chat_flow

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)

app: Flask = create_app()
executor = ThreadPoolExecutor(max_workers=4)

@app.post("/api/report")
def api_report():
    data = request.get_json(force=True) or {}
    patient = data.get("patient", {})
    features = data.get("features", {})
    deps_r = current_app.config["REPORTING_DEPS"]
    deps_rl = current_app.config["RL_DEPS"]
    deps_ins = current_app.config.get("INSIGHTS_DEPS")

    payload = {
        "id": patient.get("id", "-1"),
        "name": patient.get("name", ""),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "features": features,
    }
    result = call_reporting_flow(payload, deps_r)

    executor.submit(call_rl_flow_from_payload, deps_rl, patient, features)
    if deps_ins is not None:
        executor.submit(call_insights_flow, payload, deps_ins)

    return jsonify(result), 200

@app.post("/api/rl")
def api_rl():
    data = request.get_json(force=True) or {}
    patient = data.get("patient", {})
    features = data.get("features", {})
    deps = current_app.config["RL_DEPS"]
    executor.submit(call_rl_flow_from_payload, deps, patient, features)
    return jsonify({"status": "accepted"}), 202

@app.post("/api/insights")  
def api_insights():
    data = request.get_json(force=True) or {}
    patient = data.get("patient", {})
    features = data.get("features", {})
    deps_ins = current_app.config["INSIGHTS_DEPS"]
    payload = {
        "id": patient.get("id", "-1"),
        "name": patient.get("name", ""),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "features": features,
    }
    out = call_insights_flow(payload, deps_ins)
    return jsonify(out), 200

@app.post("/api/run")
def api_run():
    data = request.get_json(force=True) or {}
    patient = data.get("patient", {})
    features = data.get("features", {})
    deps_r = current_app.config["REPORTING_DEPS"]
    deps_rl = current_app.config["RL_DEPS"]
    deps_ins = current_app.config.get("INSIGHTS_DEPS")

    executor.submit(call_rl_flow_from_payload, deps_rl, patient, features)

    payload = {
        "id": patient.get("id", "-1"),
        "name": patient.get("name", ""),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "features": features,
    }
    if deps_ins is not None:
        executor.submit(call_insights_flow, payload, deps_ins)

    result = call_reporting_flow(payload, deps_r)
    return jsonify({"rl_dispatched": True, **result}), 200

@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True) or {}
    patient = data.get("patient", {})
    features = data.get("features")  # optional
    report   = data.get("report")    # optional (send the one you just rendered)
    deps     = current_app.config["CHAT_DEPS"]
    pid      = patient.get("id", "-1")
    msg      = data.get("message", "")

    out = call_chat_flow(deps, pid, msg, features=features, report=report)
    return jsonify(out), 200

