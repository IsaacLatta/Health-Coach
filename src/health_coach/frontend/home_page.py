import os
import json
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

try:
    import requests
except Exception:
    requests = None

st.set_page_config(
    page_title="Agent Health Risk Prototype",
    page_icon="🩺",
    layout="wide",
)

BADGE_CSS = """
<style>
.badge {display:inline-block;padding:0.35rem 0.6rem;border-radius:999px;font-weight:600;font-size:0.85rem;color:white}
.badge-low{background:#16a34a}
.badge-medium{background:#f59e0b}
.badge-high{background:#dc2626}
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)

def _clean_url(s: str) -> str:
    # strip accidental quotes from .env like "http://127.0.0.1:8000"
    return (s or "").strip().strip('"').strip("'")

DEFAULT_BACKEND_URL = _clean_url(os.environ.get("BACKEND_URL", "http://localhost:8000"))

SESSION_KEYS = {
    "features": [],
    "backend_url": DEFAULT_BACKEND_URL,
    "demo_mode": False,
    "last_run": None,
    # chat state (added)
    "chat_history": [],
    "chat_open": False,
}
for key, default in SESSION_KEYS.items():
    if key not in st.session_state:
        st.session_state[key] = default

def backend_url() -> str:
    return _clean_url(st.session_state.get("backend_url", DEFAULT_BACKEND_URL)).rstrip("/")

def risk_badge(label: str) -> str:
    cls = {"low": "badge-low", "medium": "badge-medium", "high": "badge-high"}.get(
        str(label).lower(), "badge-medium"
    )
    return f'<span class="badge {cls}">{label.title()}</span>'

def post_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    if st.session_state.get("demo_mode", False) or requests is None:
        return mock_run_response(payload)
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def mock_run_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, lightweight mock for front-end dev without backend.
    Returns structure compatible with /api/report.
    """
    patient = payload.get("patient", {})
    feats = payload.get("features", {})
    numeric_vals = [
        float(v) for v in feats.values()
        if isinstance(v, (int, float)) or str(v).replace(".", "", 1).isdigit()
    ]
    avg = sum(numeric_vals) / len(numeric_vals) if numeric_vals else 0.42
    prob = max(0.01, min(0.99, (avg % 1.0)))
    label = "low" if prob < 0.33 else ("medium" if prob < 0.66 else "high")

    shap_items = []
    for i, (k, v) in enumerate(list(feats.items())[:5]):
        phi = round(((hash(k) % 100) / 100.0 - 0.5) * 0.2, 3)
        shap_items.append({
            "feature": k,
            "value": v,
            "phi": phi,
            "reason": f"{k} influenced risk by {phi:+}."
        })
    if not shap_items:
        shap_items = [
            {"feature": "placeholder_feature", "value": 1, "phi": 0.07, "reason": "Example contribution."}
        ]

    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    html = f"""
    <div style='font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif;padding:16px'>
      <h2 style='margin-top:0'>Patient Risk Report</h2>
      <p><strong>Patient:</strong> {patient.get('name','Unknown')} &nbsp; • &nbsp;
         <strong>ID:</strong> {patient.get('id','-')} &nbsp; • &nbsp;
         <strong>Age:</strong> {patient.get('age','-')} &nbsp; • &nbsp;
         <strong>Generated:</strong> {created_at}</p>
      <hr/>
      <h3>Overall Risk</h3>
      <p>Probability: <strong>{prob:.1%}</strong> &nbsp; Label: <strong>{label.title()}</strong></p>
      <h3>Key Factors</h3>
      <ul>
        {''.join([f"<li><strong>{it['feature']}</strong>: {it['reason']}</li>" for it in shap_items])}
      </ul>
      <p style='color:#64748b'>This is a prototype report. For demo purposes only.</p>
    </div>
    """
    return {
        "run_id": run_id,
        "risk": {"prob": prob, "label": label},
        "shap": shap_items,
        "report": {"html": html, "created_at": created_at},
    }

SAMPLE_PATIENT = {
    "id": "P-0002",
    "name": "Jane Doe",
    "age": 54,
    "gender": "Female",
}
SAMPLE_FEATURES = {
    "systolic_bp": 138,
    "diastolic_bp": 86,
    "bmi": 29.4,
    "smoker": "no",
    "hdl": 42,
    "ldl": 131,
    "family_history": "yes",
    "activity_minutes_per_week": 60,
}

st.sidebar.header("Settings")
st.sidebar.text_input("Backend URL", value=st.session_state["backend_url"], key="backend_url")
st.sidebar.checkbox("Demo mode (no backend)", value=st.session_state["demo_mode"], key="demo_mode")

with st.sidebar.expander("Hooks / Extensibility", expanded=False):
    st.markdown("- Agent chat (coming soon)\n- Patient history & trends\n- PDF export\n- Auth (role: doctor)")

st.title("🫀 Agent Advisor")
st.caption("Demo UI: doctor enters patient details + features → backend runs risk report + agentic RL (config tuning), then returns a readable report.")

with st.container():
    st.subheader("Patient Intake")
    col1, col2, col3, col4 = st.columns([1.4, 1, 1, 1])
    with col1:
        p_name = st.text_input("Full name", placeholder="e.g., Jane Doe")
    with col2:
        p_id = st.text_input("Patient ID", placeholder="e.g., P-00123")
    with col3:
        p_age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
    with col4:
        p_gender = st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])

    st.markdown("\n")
    with st.expander("Patient Features (model inputs)", expanded=True):
        st.caption("These match the regression model’s expected features.")

        colA, colB, colC = st.columns(3)

        with colA:
            f_age = st.number_input("Age", min_value=1, max_value=120, value=54, step=1)
            f_sex = st.selectbox("Sex", ["Female", "Male"], index=0)
            f_bp = st.number_input("Resting Blood Pressure (BP)", min_value=60, max_value=240, value=130)
            f_chol = st.number_input("Cholesterol", min_value=80, max_value=400, value=220)

        with colB:
            f_fbs = st.text_input("Fasting Blood Sugar", value="no", help="yes/no or numeric; >120 → yes")
            f_restecg = st.selectbox("Resting ECG (0–2)", [0,1,2], index=1)
            f_thalach = st.number_input("Max HR (Thalach)", min_value=60, max_value=250, value=160)
            f_exang = st.selectbox("Exercise Angina", ["no","yes"], index=0)

        with colC:
            f_oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
            f_cp = st.selectbox("Chest Pain Type", ["typical","atypical","non-anginal","asymptomatic"], index=3)
            f_slope = st.selectbox("Slope of ST", ["downsloping","flat","upsloping"], index=1)
            f_ca = st.selectbox("Number of vessels (Ca)", [0,1,2,3], index=0)
            f_thal = st.selectbox("Thallium", ["normal","fixed defect","reversible defect"], index=0)

        model_features = {
            "Age": f_age,
            "Sex": f_sex,
            "BP": f_bp,
            "Cholesterol": f_chol,
            "FBS over 120": f_fbs,
            "EKG results": f_restecg,
            "Max HR": f_thalach,
            "Exercise angina": f_exang,
            "ST depression": f_oldpeak,
            "Chest pain type": f_cp,
            "Slope of ST": f_slope,
            "Number of vessels fluro": f_ca,
            "Thallium": f_thal,
        }

        if st.button("Add feature"):
            st.session_state.features.append({"name": "", "value": ""})

    st.markdown("\n")
    run_col, reset_col = st.columns([1, 1])
    with run_col:
        run_clicked = st.button("▶️ Run Assessment", type="primary")
    with reset_col:
        if st.button("Reset form"):
            st.session_state.features = []
            st.session_state.chat_history = []
            st.session_state.chat_open = False
            st.session_state.last_run = None
            st.experimental_rerun()

result: Dict[str, Any] = {}
if run_clicked:
    payload = {"patient": SAMPLE_PATIENT, "features": SAMPLE_FEATURES}
    with st.spinner("Running risk report…"):
        try:
            result = post_json(f"{backend_url()}/api/report", payload)
            st.session_state.last_run = result
        except Exception as e:
            st.error(f"Failed to run assessment: {e}")

show = result or st.session_state.get("last_run")
if show:
    st.markdown("---")
    st.subheader("Risk Report")

    prob = show.get("risk", {}).get("prob", None)
    label = show.get("risk", {}).get("label", "")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Probability", f"{(prob or 0.0):.1%}" if prob is not None else "–")
    with c2:
        st.markdown(risk_badge(label), unsafe_allow_html=True)
    with c3:
        st.progress(min(1.0, max(0.0, float(prob or 0.0))))

    shap_items = show.get("shap", [])
    if shap_items:
        st.markdown("**Top factors**")
        df = pd.DataFrame(shap_items)
        cols = [c for c in ["feature", "value", "phi", "reason"] if c in df.columns]
        df = df[cols]
        dtype_map = {}
        if "feature" in df.columns: dtype_map["feature"] = "string"
        if "reason"  in df.columns: dtype_map["reason"]  = "string"
        if "value"   in df.columns: dtype_map["value"]   = "string"
        if "phi"     in df.columns: dtype_map["phi"]     = "float64"
        if dtype_map:
            df = df.astype(dtype_map).fillna("")
        st.dataframe(df, use_container_width=True, hide_index=True)

    report = show.get("report", {})
    html = report.get("html", "")
    if html:
        with st.expander("View full report", expanded=True):
            components.html(html, height=420, scrolling=True)

        out_id = SAMPLE_PATIENT["id"]
        fname = f"risk_report_{out_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.html"
        st.download_button(label="⬇️ Download report (HTML)", data=html, file_name=fname, mime="text/html")

    exp_open = st.session_state.chat_open or bool(st.session_state.chat_history)
    with st.expander("Agent chat", expanded=exp_open):
        for turn in st.session_state.chat_history:
            role = turn.get("role", "assistant")
            text = turn.get("text", "")
            with st.chat_message(role):
                st.markdown(text)

        user_msg = st.chat_input("Ask about this patient's risk, drivers, or RL behavior…")
        if user_msg:
            st.session_state.chat_open = True  # keep panel open on subsequent reruns
            st.session_state.chat_history.append({"role": "user", "text": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            try:
                payload = {
                    "patient": SAMPLE_PATIENT,
                    "features": SAMPLE_FEATURES,
                    "report": show,    # send current report context to backend
                    "message": user_msg,
                }
                resp = post_json(f"{backend_url()}/api/chat", payload)
                reply = resp.get("reply", {}) if isinstance(resp, dict) else {}
                text = (
                    (reply.get("answer") if isinstance(reply, dict) else None)
                    or (resp.get("text") if isinstance(resp, dict) else None)
                    or "No answer."
                )
                bullets = reply.get("bullets", []) if isinstance(reply, dict) else []
                used = reply.get("used_signals", []) if isinstance(reply, dict) else []

                st.session_state.chat_history.append({"role": "assistant", "text": text})
                with st.chat_message("assistant"):
                    st.markdown(text)
                    for b in bullets:
                        st.markdown(f"- {b}")
                    if used:
                        st.caption("Signals used: " + ", ".join(str(u) for u in used))
            except Exception as e:
                err = f"Chat failed: {e}"
                st.session_state.chat_history.append({"role": "assistant", "text": err})
                with st.chat_message("assistant"):
                    st.error(err)

    with st.expander("History & trends (coming soon)"):
        st.info("We will show recent visits, risk trajectory, and action/reward transitions.")

    with st.expander("RL Insights (agent)", expanded=False):
        try:
            insights = post_json(f"{backend_url()}/api/insights", {
                "patient": SAMPLE_PATIENT,
                "features": SAMPLE_FEATURES,
            })
            metrics = insights.get("metrics", {})
            summary = insights.get("summary", {})

            cols = st.columns(3)
            with cols[0]:
                st.metric("Margin → moderate", f"{metrics.get('margin_to_moderate', 0.0):+.1%}")
            with cols[1]:
                st.metric("Margin → high", f"{metrics.get('margin_to_high', 0.0):+.1%}")
            with cols[2]:
                vol = metrics.get("volatility", "low")
                vv  = metrics.get("volatility_value", 0.0)
                st.metric("Volatility", f"{vol} ({vv:.3f})")

            action_stats = metrics.get("action_stats", [])
            if action_stats:
                st.markdown("**Action → reward (recent)**")
                df_actions = pd.DataFrame(action_stats)
                order = [c for c in ["action", "count", "mean_reward"] if c in df_actions.columns]
                df_actions = df_actions[order].astype({"action":"int64","count":"int64","mean_reward":"float64"})
                st.dataframe(df_actions, use_container_width=True, hide_index=True)

            if summary.get("narrative"):
                st.markdown("**Narrative**")
                st.write(summary["narrative"])
            if summary.get("bullets"):
                st.markdown("**Bullets**")
                for b in summary["bullets"]:
                    st.markdown(f"- {b}")
        except Exception as e:
            st.info(f"Insights not available yet. ({e})")

st.markdown("<hr style='opacity:0.2'/>", unsafe_allow_html=True)
st.caption("Prototype for demo purposes — the RL system tunes reporting parameters server-side; clinicians see the report only.")
