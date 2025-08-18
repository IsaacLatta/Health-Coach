from abc import abstractmethod, ABC
from typing import Any, Dict, List

class TemplateService(ABC):
    @abstractmethod
    def render(self, info: Any) -> str: ...

class SimpleHTMLTemplate(TemplateService):
    def render(self, info: Dict[str, Any]) -> str:
        patient = info.get("patient", {})
        risk = info.get("risk", {})
        shap: List[Dict[str, Any]] = info.get("shap", [])
        items = "".join(
            f"<li><strong>{i['feature']}</strong>: {i.get('reason','')}</li>" for i in shap
        )
        return f"""
        <div style='font-family:system-ui;padding:16px'>
          <h2>Patient Risk Report</h2>
          <p><b>Patient:</b> {patient.get('name','-')} &nbsp; • &nbsp;
             <b>ID:</b> {patient.get('id','-')} &nbsp; • &nbsp;
             <b>Age:</b> {patient.get('age','-')}</p>
          <hr/>
          <p><b>Probability:</b> {risk.get('prob',0):.1%} &nbsp; 
             <b>Label:</b> {risk.get('label','-').title()}</p>
          <h3>Key Factors</h3>
          <ul>{items or '<li>–</li>'}</ul>
          <p style='color:#64748b'>Prototype output (mock).</p>
        </div>
        """
