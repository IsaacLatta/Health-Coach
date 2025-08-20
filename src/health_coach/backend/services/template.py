from abc import abstractmethod, ABC
from typing import Any, Dict, List

class TemplateService(ABC):
    @abstractmethod
    def render(self, info: Any) -> str: ...

class SimpleHTMLTemplate(TemplateService):
    def render(self, info: Dict[str, Any]) -> str:
        patient = info.get("patient", {})
        risk = info.get("risk", {})
        shap_raw: List[Dict[str, Any]] = info.get("shap", []) or []

        explanations = (info.get("explanations", {}) or {}).get("items", [])
        summary = info.get("summary", {}) or {}

        if explanations:
            items_html = "".join(
                f"<li><strong>{it.get('feature','')}</strong> "
                f"(value: {it.get('value','–')}, φ={it.get('phi','–')}, "
                f"{it.get('direction','?')} by {it.get('magnitude','–')}): "
                f"{it.get('explanation','')}</li>"
                for it in explanations
            )
        else:
            items_html = "".join(
                f"<li><strong>{i.get('feature','')}</strong>: {i.get('reason','')}</li>"
                for i in shap_raw
            ) or "<li>–</li>"

        narrative_html = ""
        if summary:
            narrative_html = (
                f"<h3>Overall Summary</h3>"
                f"<p>{summary.get('narrative','')}</p>"
            )

        return f"""
        <div style='font-family:system-ui;padding:16px'>
          <h2>Patient Risk Report</h2>
          <p><b>Patient:</b> {patient.get('name','-')} &nbsp; • &nbsp;
             <b>ID:</b> {patient.get('id','-')} &nbsp; • &nbsp;
             <b>Age:</b> {patient.get('age','-')}</p>
          <hr/>
          <p><b>Probability:</b> {float(risk.get('prob',0)):.1%} &nbsp; 
             <b>Label:</b> {str(risk.get('label','-')).title()}</p>
          {narrative_html}
          <h3>Key Factors</h3>
          <ul>{items_html}</ul>
          <p style='color:#64748b'>Prototype output. Decision support only.</p>
        </div>
        """

class VerboseTemplate(TemplateService):
    def render(self, info: Dict[str, Any]) -> str:
        patient = info.get("patient", {}) or {}
        risk = info.get("risk", {}) or {}
        summary = info.get("summary", {}) or {}
        explanations = (info.get("explanations", {}) or {}).get("items", []) or []
        shap_raw: List[Dict[str, Any]] = info.get("shap", []) or []
        # features for the table: each item should have: name, value, phi, contrib
        features: List[Dict[str, Any]] = info.get("features", []) or []

        # If caller didn't supply 'features', derive from explanations or shap_raw
        if not features:
            src = explanations if explanations else shap_raw
            for it in src:
                features.append({
                    "name": it.get("feature",""),
                    "value": it.get("value",""),
                    "phi": it.get("phi", 0.0),
                    "contrib": it.get("direction", "increases") if "direction" in it
                               else ("increases" if float(it.get("phi",0)) >= 0 else "decreases"),
                })

        prob = float(risk.get("prob", 0.0))
        percent = f"{prob*100:.1f}"
        spread = 10 - abs(prob - 0.5) * 20
        low = max(0.0, (prob*100 - spread/2))
        high = min(100.0, (prob*100 + spread/2))
        conf_range = f"{low:.1f}% – {high:.1f}%"
        level = "High" if prob >= 0.66 else ("Medium" if prob >= 0.33 else "Low")

        def esc(x: Any) -> str:
            return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

        rows_html = "\n".join(
            f"<tr><td>{esc(f.get('name',''))}</td>"
            f"<td>{esc(f.get('value',''))}</td>"
            f"<td>—</td>"
            f"<td>{esc(f.get('phi',''))}</td>"
            f"<td>{esc(f.get('contrib',''))}</td></tr>"
            for f in features
        )

        narrative_html = ""
        if summary.get("narrative"):
            narrative_html = f"""
            <div class="section">
              <h2>Interpretation</h2>
              <div class="interpretation-box">
                {esc(summary.get("narrative",""))}
              </div>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Medical Assessment Report</title>
  <style>
    body {{
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background-color: #f4f6f9;
      color: #333;
      margin: 30px;
      line-height: 1.6;
    }}
    h1, h2 {{ color: #2c3e50; margin-bottom: 10px; }}
    .header-note {{ font-size: 0.95em; font-style: italic; color: #888; margin-bottom: 20px; }}
    .summary-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 40px; }}
    .summary-box, .probability-box {{
      background-color: #fff; padding: 20px; border-radius: 6px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.05); flex: 1 1 300px;
    }}
    .summary-box {{ border-left: 5px solid #3498db; }}
    .summary-grid {{ display: flex; justify-content: space-around; gap: 40px; text-align: center; }}
    .summary-grid div span {{ display: block; font-weight: bold; color: #2980b9; }}
    .probability-box {{ border: 2px solid #ffb84d; background-color: #fff3e0; text-align: center; }}
    .probability-box .percentage {{ font-size: 2em; color: #c0392b; font-weight: bold; }}
    .confidence-info {{ margin-top: 15px; font-size: 0.95em; color: #444; }}
    .confidence-level-indicator {{ display: flex; justify-content: center; gap: 20px; margin-top: 10px; }}
    .level-box {{ padding: 6px 14px; border: 2px solid #bbb; border-radius: 4px; font-weight: 500; color: #555; }}
    .level-box.Low.active {{ background-color: #f8d7da; border-color: #dc3545; color: #721c24; }}
    .level-box.Medium.active {{ background-color: #fff3cd; border-color: #ffc107; color: #856404; }}
    .level-box.High.active {{ background-color: #d4edda; border-color: #28a745; color: #155724; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0 8px; }}
    thead th {{ background-color: #2980b9; color: #fff; padding: 10px; text-align: left; }}
    tbody td {{ background-color: #fff; padding: 10px; border-bottom: 1px solid #ddd; }}
    tbody tr:hover td {{ background-color: #f0f8ff; }}
    .section {{ margin-bottom: 40px; }}
    .interpretation-box {{ background-color: #f9f9f9; padding: 20px; border-left: 4px solid #7f8c8d; font-size: 1.05em; border-radius: 4px; }}
    .final-note {{ font-style: italic; color: #666; font-size: 0.9em; margin-top: 40px; }}
    hr {{ border: 0; border-top: 1px solid #ccc; margin: 40px 0; }}
    @media (max-width: 768px) {{ .summary-grid {{ flex-direction: column; align-items: center; }} }}
  </style>
</head>
<body>
  <h1>Medical Assessment Report</h1>
  <p class="header-note">This report is generated by AI Agents and should be interpreted only by qualified doctors.</p>

  <div class="summary-container">
    <div class="summary-box">
      <h2>Patient Summary</h2>
      <div class="summary-grid">
        <div><span>Patient ID</span>{esc(patient.get('id','-'))}</div>
        <div><span>Age</span>{esc(patient.get('age','-'))}</div>
        <div><span>Sex</span>{esc(patient.get('gender', patient.get('sex','-')))}</div>
      </div>
    </div>

    <div class="probability-box">
      <h2>Predicted Heart Disease Probability</h2>
      <p class="percentage">{percent}%</p>
      <div class="confidence-info">
        <p><strong>Uncertainty Range:</strong> {conf_range}</p>
        <p><strong>Confidence Level:</strong></p>
        <div class="confidence-level-indicator">
          <div class="level-box Low {'active' if level=='Low' else ''}">Low</div>
          <div class="level-box Medium {'active' if level=='Medium' else ''}">Medium</div>
          <div class="level-box High {'active' if level=='High' else ''}">High</div>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Physiological Parameters</h2>
    <table>
      <thead>
        <tr>
          <th>Parameter</th><th>Value</th><th>Reference Interval</th><th>SHAP Value</th><th>Contribution</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  {narrative_html}

  <p class="final-note">Disclaimer: This report is intended strictly for use by licensed healthcare professionals. AI-generated outputs are not a substitute for clinical judgment.</p>
</body>
</html>"""
