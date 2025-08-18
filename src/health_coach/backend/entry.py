from flask import Flask, request

from .agents.flows.reporting.reporting_flow import call_reporting_flow
from .agents.flows.rl.rl_flow import call_rl_flow

app = Flask(__name__)

@app.route("/api/run", methods=["POST"])
async def run():
    try:

        json_data = request.get_json()
        patient_data = json_data.get('patient', {})
        if patient_data is None:
            return 400

        call_rl_flow(patient_data)
        report = await call_reporting_flow(patient_data)
        # serve the report back to the client
        return 200
    except Exception as e:
        return 500