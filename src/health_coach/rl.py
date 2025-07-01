from typing import Dict
from abc import ABC, abstractmethod
from crewai import Crew, Process

import health_coach.tasks as tasks
import health_coach.agents as agents

import health_coach.tools.rl as rl_tools
import health_coach.tools.data as data_tools

class RLImplementation(ABC):
    @abstractmethod
    def create_crew(self) -> Crew:
        pass

class QLearningImplementation(RLImplementation):
    def create_crew(self) -> Crew:
        data_loader = agents.DataLoaderAgent().create()
        policy_agent = agents.PolicyAgent().create()
        reward_shaping_agent = agents.RewardShapingAgent().create()
        data_exporter = agents.DataExportAgent().create()
        
        t0 = tasks.FetchPatientHistory().create(data_loader, tools=[data_tools.load_patient_history])
        t1 = tasks.ComputeCurrentState().create(policy_agent, tools=[rl_tools.discretize_probability])
        t2 = tasks.ComputeReward().create(reward_shaping_agent, tools=[rl_tools.compute_reward])
        t3 = tasks.ComputeAction().create(policy_agent, tools=[rl_tools.select_action])
        t4 = tasks.ShapeReward().create(reward_shaping_agent, tools=[rl_tools.amplify_reward], context=[t0, t2])
        t5 = tasks.UpdateRLModel().create(data_exporter, tools=[rl_tools.update_rl_model])

        return Crew(agents=[data_loader, reward_shaping_agent, policy_agent, data_exporter],
                    tasks=[t0, t1, t2, t3, t4, t5],
                    process=Process.sequential,
                    verbose=False)
        