from pydantic import BaseModel
from typing import Any, List, Dict, Optional
from pathlib import Path
from crewai.flow.flow import Flow, listen, start

from health_coach.rl import QLearningImplementation, RLImplementation
import health_coach.tasks as tasks
import health_coach.agents as agents
import health_coach.tools.data as data_tools
import health_coach.tools.rl as rl_tools
import health_coach.tools.prediction as pred_tools
class RLInput(BaseModel):
    patient_id: str = ""
    patient_features: List[float] = None

class RLState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    context_summary: str = ""
    input: RLInput = 0
    reward: float = 0.0
    env_reward: float = 0.0
    current_state: int = 0
    previous_state: int = 0
    action: int = 0
    rl_implementation: Optional[RLImplementation] = None

class RLFlow(Flow[RLState]):
    def set_rl_implementation(self, impl: RLImplementation):
        self.state.rl_implementation = impl

    def set_input(self, input: RLInput):
        self.state.input = input
    
    @start()
    def load_data(self):
        patient_info = data_tools._load_patient_data(self.state.input.patient_id, timeout_base=0,)
        self.state.previous_state = patient_info["State"]
        return
    
    @listen(load_data)
    def compute_current_state(self):
        prediction = pred_tools._make_prediction(self.state.input.patient_features)
        self.state.current_state = rl_tools._discretize_probability(prediction)
        self.state.env_reward = rl_tools._compute_reward(self.state.previous_state, self.state.current_state)
        return 
    
    @listen(compute_current_state)
    def generate_shaping_contexts(self):
        inputs = {
            "pure_reward" : self.state.env_reward,
            "current_state": self.state.current_state,
            "previous_state": self.state.previous_state,
            "possible_actions": [0, 1, 2, 3, 4, 5, 6], # hardcode for now
        }
        print(f"Input for context pipeline: {inputs}")

        crew = self.state.rl_implementation.create_context_crew()
        context = "Unavailable"
        try:
            result = crew.kickoff(inputs=inputs)
            raw = result.json_dict.get("context", None)
            if isinstance(raw, str) and raw.strip():
                context = raw
        except:
            pass
        self.state.context_summary = context
        print(f"Final shaping context: {self.state.context_summary}")
        return
    
    @listen(generate_shaping_contexts)
    def shape_reward_and_action(self):
        inputs = {
            "current_state": self.state.current_state,
            "previous_state": self.state.previous_state,
            "possible_actions": [0, 1, 2, 3, 4], # hardcode for now
            "context": self.state.context_summary
        }

        crew = self.state.rl_implementation.create_shaping_crew()
        try:
            action_out, reward_out = crew.kickoff(inputs=inputs).tasks_output
            print(f"\n\nAction: {action_out.raw}\nReward: {reward_out.raw}\n\n")
            self.state.action = action_out.json_dict["action"]
            self.state.reward = reward_out.json_dict["shaped_reward"]
            return
        except Exception as e:
            print(f"Error: {e}")
        return
    
    @listen(generate_shaping_contexts)
    def update_rl_model(self):
        return {
            "action": self.state.action,
            "reward" : self.state.reward,
            "previous_state": self.state.previous_state,
            "current_state": self.state.current_state,
            }
