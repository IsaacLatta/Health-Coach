from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel

from crewai.tasks.hallucination_guardrail import HallucinationGuardrail
from crewai.knowledge.source import base_knowledge_source
from crewai import Agent, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

from health_coach.backend.flows.knowledge_sources import get_all_sources

from health_coach.backend.flows.llm import get_embedder, get_llm

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
            kwargs["llm"] = llm

        embedder = overrides.get("embedder", self.default_embedder())
        if embedder is not None:
            kwargs["embedder"] = embedder

        return Agent(**kwargs)

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
        return get_all_sources()
    
    def default_verbose(self) -> bool:
        return True

    def default_llm(self):
        return None
        # return get_llm()
    
    def default_embedder(self):
        return None
        # return get_embedder()