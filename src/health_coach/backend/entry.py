# entry.py
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, current_app
from .app import create_app
from .flows.reporting.reporting_flow import call_reporting_flow
from .flows.rl.rl_flow import call_rl_flow  # kicks off RL step

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)

app: Flask = create_app()
executor = ThreadPoolExecutor(max_workers=4)

@app.post("/api/report")
def api_report():
    data = request.get_json(force=True) or {}
    patient = data.get("patient", {})
    features = data.get("features", {})
    deps = current_app.config["REPORTING_DEPS"]

    payload = {
        "id": patient.get("id", "-1"),
        "name": patient.get("name", ""),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "features": features,
    }
    result = call_reporting_flow(payload, deps)
    return jsonify(result), 200

@app.post("/api/rl")
def api_rl():
    data = request.get_json(force=True) or {}
    patient = data.get("patient", {})
    deps = current_app.config["RL_DEPS"]
    executor.submit(call_rl_flow, patient, deps)
    return jsonify({"status": "accepted"}), 202

@app.post("/api/run")
def api_run():
    data = request.get_json(force=True) or {}
    deps_r = current_app.config["REPORTING_DEPS"]
    deps_rl = current_app.config["RL_DEPS"]
    executor.submit(call_rl_flow, data.get("patient", {}), deps_rl)
    result = call_reporting_flow({
        "id": data.get("patient", {}).get("id", "-1"),
        "name": data.get("patient", {}).get("name", ""),
        "age": data.get("patient", {}).get("age"),
        "gender": data.get("patient", {}).get("gender"),
        "features": data.get("features", {}),
    }, deps_r)
    return jsonify({"rl_dispatched": True, **result}), 200
