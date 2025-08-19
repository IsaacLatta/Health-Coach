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
