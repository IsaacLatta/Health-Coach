import os
import json
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
SESSION_KEYS = {
    "features": [],
    "backend_url": DEFAULT_BACKEND_URL,
    "demo_mode": False,
    "last_run": None,
}

for key, default in SESSION_KEYS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def backend_url() -> str:
    return st.session_state.get("backend_url", DEFAULT_BACKEND_URL).rstrip("/")


def risk_badge(label: str) -> str:
    cls = {
        "low": "badge-low",
        "medium": "badge-medium",
        "high": "badge-high",
    }.get(str(label).lower(), "badge-medium")
    return f'<span class="badge {cls}">{label.title()}</span>'


def post_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    if st.session_state.get("demo_mode", False) or requests is None:
        return mock_run_response(payload)
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def mock_run_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, lightweight mock for front-end dev without backend.
    Returns structure compatible with /api/run or /api/report.
    """
    patient = payload.get("patient", {})
    feats = payload.get("features", {})
    numeric_vals = [float(v) for v in feats.values()
                    if isinstance(v, (int, float)) or str(v).replace('.', '', 1).isdigit()]
    avg = sum(numeric_vals)/len(numeric_vals) if numeric_vals else 0.42
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
    created_at = datetime.utcnow().isoformat() + "Z"
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


# --------- FIXED MOCK PAYLOAD FOR SERVER TESTING ---------
SAMPLE_PATIENT = {
    "id": "P-0001",
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
# ---------------------------------------------------------


st.sidebar.header("Settings")
st.sidebar.text_input("Backend URL", value=st.session_state["backend_url"], key="backend_url")
st.sidebar.checkbox("Demo mode (no backend)", value=st.session_state["demo_mode"], key="demo_mode")

with st.sidebar.expander("Hooks / Extensibility", expanded=False):
    st.markdown("- Agent chat (coming soon)\n- Patient history & trends\n- PDF export\n- Auth (role: doctor)")

st.title("🩺 Agent Health Risk — Prototype")
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
    with st.expander("Patient Features (add as needed)", expanded=True):
        st.caption("Add any model inputs here. Leave blank if unknown. You can add new rows dynamically.")

        if not isinstance(st.session_state.features, list):
            st.session_state.features = []

        remove_indices: List[int] = []
        for idx, row in enumerate(st.session_state.features):
            c1, c2, c3 = st.columns([1.2, 1.4, 0.3])
            with c1:
                name = st.text_input(
                    f"Feature name {idx}",
                    value=row.get("name", ""),
                    key=f"feat_name_{idx}",
                    placeholder="e.g., systolic_bp",
                )
            with c2:
                value = st.text_input(
                    f"Feature value {idx}",
                    value=str(row.get("value", "")),
                    key=f"feat_value_{idx}",
                    placeholder="e.g., 126",
                )
            with c3:
                if st.button("🗑️", key=f"feat_del_{idx}"):
                    remove_indices.append(idx)

            st.session_state.features[idx] = {"name": name, "value": value}

        if remove_indices:
            st.session_state.features = [r for i, r in enumerate(st.session_state.features) if i not in remove_indices]

        if st.button("Add feature"):
            st.session_state.features.append({"name": "", "value": ""})

    st.markdown("\n")
    run_col, reset_col = st.columns([1, 1])
    with run_col:
        run_clicked = st.button("▶️ Run Assessment", type="primary")
    with reset_col:
        if st.button("Reset form"):
            st.session_state.features = []
            st.experimental_rerun()

result: Dict[str, Any] = {}
if run_clicked:
    payload = {
        "patient": SAMPLE_PATIENT,
        "features": SAMPLE_FEATURES,
    }
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
        if prob is not None:
            st.metric("Probability", f"{prob:.1%}")
        else:
            st.metric("Probability", "–")
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
        st.dataframe(df, use_container_width=True, hide_index=True)

    report = show.get("report", {})
    html = report.get("html", "")
    if html:
        with st.expander("View full report", expanded=True):
            components.html(html, height=420, scrolling=True)

        out_id = SAMPLE_PATIENT["id"]
        fname = f"risk_report_{out_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.html"
        st.download_button(
            label="⬇️ Download report (HTML)",
            data=html,
            file_name=fname,
            mime="text/html",
        )

    with st.expander("Agent chat (coming soon)"):
        st.info("Chat agent will summarize trends and justify configuration decisions. (Not wired yet.)")
        st.text_input("Message", placeholder="Ask about this patient's trend…", disabled=True)

    with st.expander("History & trends (coming soon)"):
        st.info("We will show recent visits, risk trajectory, and action/reward transitions.")

st.markdown("<hr style='opacity:0.2'/>", unsafe_allow_html=True)
st.caption("Prototype for demo purposes — the RL system tunes reporting parameters server-side; clinicians see the report only.")
