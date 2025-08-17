from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel

from crewai.tasks.hallucination_guardrail import HallucinationGuardrail
from crewai.knowledge.source import base_knowledge_source
from crewai import Agent, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

from health_coach.tools.rl_tools.knowledge import get_all_sources

import health_coach.config as cfg

class AgentProxy(ABC):
    @abstractmethod
    def default_name(self) -> str:
        ...

    @abstractmethod
    def default_role(self) -> str:
        ...

    @abstractmethod
    def default_goal(self) -> str:
        ...

    @abstractmethod
    def default_backstory(self) -> str:
        ...

    def default_knowledge_sources(self) -> Optional[List[base_knowledge_source.BaseKnowledgeSource]]:
        return None

    def default_max_iter(self) -> Optional[int]:
        return 1

    def default_verbose(self) -> Optional[bool]:
        return True
    
    def default_llm(self) -> Optional[Any]:
        return None

    def default_embedder(self) -> Optional[Any]:
        return None

    def create(self, **overrides: Any) -> Agent:
        kwargs: Dict[str, Any] = {}
        name = overrides.get("name", self.default_name())
        if name is not None:
            kwargs["name"] = name
        role = overrides.get("role", self.default_role())
        if role is not None:
            kwargs["role"] = role
        goal = overrides.get("goal", self.default_goal())
        if goal is not None:
            kwargs["goal"] = goal
        backstory = overrides.get("backstory", self.default_backstory())
        if backstory is not None:
            kwargs["backstory"] = backstory
        max_iter = overrides.get("max_iter", self.default_max_iter())
        if max_iter is not None:
            kwargs["max_iter"] = max_iter
        verbose = overrides.get("verbose", self.default_verbose())
        if verbose is not None:
            kwargs["verbose"] = verbose
        knowledge_sources = overrides.get("knowledge_sources", self.default_knowledge_sources())
        if knowledge_sources is not None:
            kwargs["knowledge_sources"] = knowledge_sources
        llm = overrides.get("llm", self.default_llm())
        if llm is not None:
            print("\n\nLLM found!\n\n")
            kwargs["llm"] = llm
        else:
            print("\n\nNo LLM found.\n\n")

        embedder = overrides.get("embedder", self.default_embedder())
        if embedder is not None:
            print(f"\n\nEmbedder found: {embedder}\n\n")
            kwargs["embedder"] = embedder
        else:
            print("\n\nEmbedder not found!\n\n")

        return Agent(**kwargs)

class PolicyAgent(AgentProxy):
    def default_name(self) -> str:
        return "policy_agent"

    def default_role(self) -> str:
        return (
            "Your responsibility is to act as the policy oracle within the "
            "reinforcement learning loop, selecting the next discrete action index."
        )

    def default_goal(self) -> str:
        return "Given a state identifier, select the next action index."

    def default_backstory(self) -> str:
        return (
            "You are the sentient embodiment of a decision-maker. Through countless simulations, "
            "you have mastered translating states into discrete actions using the select_action tool."
        )

class ContextProvidingAgent(AgentProxy):
    def default_name(self) -> str:
        return "context_providing_agent"

    def default_role(self) -> str:
        return (
            "You are the Context Providing Agent. Your responsibility is to gather and "
            "summarize relevant contextual information about a data source using the provided tools."
        )

    def default_goal(self) -> str:
        return (
            "Given a data source and access to a suite of tools, "
            "produce a concise textual explanation of the key contextual signals in JSON."
        )

    def default_backstory(self) -> str:
        return (
            "You are a domain-savvy analyst agent, skilled at interpreting raw data via tools. "
            "You know how to extract trends, counts, and summary statistics and craft them into a clear narrative."
        )

    def default_max_iter(self) -> Optional[int]:
        return 3
    
    def default_knowledge_source(self):
        return None
        # return StringKnowledgeSource(content=CONTEXT_KNOWLEDGE_SOURCE)
        # return TextFileKnowledgeSource(file_paths=["qlearning/q_learning.txt"])

class RewardShapingAgent(AgentProxy):
    def default_name(self) -> str:
        return "reward_shaping_agent"

    def default_role(self) -> str:
        return (
            "You are the **Reward Shaping Agent**, an expert module in the reinforcement-learning "
            "pipeline whose responsibility is to inspect each transition and adjust its reward. "
            "Using any context-providing tools, you evaluate the pure RL reward and transform it into a shaped reward that "
            "accelerates learning and guides the policy toward desirable behaviors."
        )

    def default_goal(self) -> str:
        return (
            "Given the **pure reward** computed by the Q-learning algorithm and access to your suite of context and shaping tools, "
            "produce an augmented reward that; encourages exploration of under-visited or novel states and transitions, "
            "incorporates temporal trends in recent rewards, rewards progress toward episode completion, normalizes and "
            "stabilizes the signal to ensure consistent updates."
        )

    def default_backstory(self) -> str:
        return (
            "You were instantiated as the “sentient embodiment” of the reward function in a healthcare-RL research framework. "
            "Forged from domain expertise and algorithmic insight, you wield specialized tools that read patient-episode data "
            "from CSVs, count state visits and transition frequencies, compute moving reward averages, and measure episode progress. " 
            "With each transition, you apply your profound understanding of novelty, momentum, and long-term objectives to reshape "
            "raw rewards—helping the learner converge faster and more robustly than pure Q-learning alone. "
        )
    
    def default_knowledge_sources(self):
        # return None
        return [get_all_sources()[0]]
    
    def default_verbose(self) -> bool:
        return True

    def default_llm(self):
        return LLM(
            model=cfg.LLM_MODEL,
            base_url=cfg.LLM_BASE_URL,
        )
    
    def default_embedder(self):
        # return None
        return {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
            }
        }