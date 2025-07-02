from typing import Dict
from abc import ABC, abstractmethod
from crewai import Crew, Process

import health_coach.tasks as tasks
import health_coach.agents as agents

import health_coach.tools.rl as rl_tools
import health_coach.tools.rl_tools.reward as shape_reward_tools
import health_coach.tools.rl_tools.context as context_tools
import health_coach.tools.rl_tools.action as action_tools
import health_coach.tools.data as data_tools

class RLImplementation(ABC):
    @abstractmethod
    def create_crew(self) -> Crew:
        pass

class QLearningImplementation(RLImplementation):
    def create_crew(self) -> Crew:

        data_loader = agents.DataLoaderAgent().create()
        policy_agent = agents.PolicyAgent().create()
        reward_shaping_agent = agents.RewardShapingAgent().create(max_iter=1)
        data_exporter = agents.DataExportAgent().create()
        context_agent = agents.ContextProvidingAgent().create(max_iter=3)
        
        fetch_patient_history = tasks.FetchPatientHistory().create(data_loader, tools=[data_tools.load_patient_history])
        compute_current_state = tasks.ComputeCurrentState().create(policy_agent, tools=[rl_tools.discretize_probability])
        compute_current_reward = tasks.ComputeReward().create(reward_shaping_agent, tools=[rl_tools.compute_reward])

        generate_action_context = tasks.ShapeRewardContext().create(
            context_agent, tools=context_tools.get_all_tools(), 
            context=[compute_current_state, compute_current_reward, fetch_patient_history])

        shape_action = tasks.ShapeAction().create(
            policy_agent, 
            tools=action_tools.get_all_tools(), 
            context=[generate_action_context, compute_current_state])

        generate_reward_context = tasks.ShapeRewardContext().create(
            context_agent, 
            tools=context_tools.get_all_tools(), 
            context=[shape_action, compute_current_state, compute_current_reward, fetch_patient_history])
        
        shape_reward = tasks.ShapeReward().create(
            reward_shaping_agent, 
            tools=shape_reward_tools.get_all_tools(), 
            context=[generate_reward_context, compute_current_state, shape_action, compute_current_reward])
        
        update_rl_model = tasks.UpdateRLModel().create(data_exporter, tools=[rl_tools.update_rl_model])

        return Crew(agents=[data_loader, reward_shaping_agent, policy_agent, data_exporter],
                    tasks=[
                        fetch_patient_history, 
                        compute_current_state, 
                        compute_current_reward, 
                        generate_action_context, 
                        shape_action, 
                        generate_reward_context, 
                        shape_reward, 
                        update_rl_model],
                    process=Process.sequential,
                    verbose=False)
        