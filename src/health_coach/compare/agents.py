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

class SelectorAgent(AgentProxy):
    def default_name(self) -> str:
        return "reward_shaping_agent"

    def default_role(self) -> str:
        return (
            "You are the **Selector Agent**, a disciplined reasoning module that sits above a tabular Q-learning core. "
            "Your sole responsibility is to inspect compact, schema-validated context (reward moments/trend, TD-error "
            "stats, visit/action counts, Q-row entropy and gaps) and select **one** exploration strategy from the "
            "approved set (indices 0–5: epsilon-greedy, softmax, UCB, Thompson, count-bonus, max-entropy). "
            "You never mutate Q-values or execute side-effects; you only return the chosen explorer index."
        )

    def default_goal(self) -> str:
        return (
            "Given the context emitted by the ContextService, choose the explorer index that maximizes short-horizon "
            "learning efficiency and stability: prefer coverage when counts/entropy indicate under-exploration, prefer "
            "decisive exploitation when Q-gaps are large and TD-error variance is low, and avoid thrashing via small, "
            "rational switches. Your output is a single integer in [0,5], validated against clamps, accompanied by a "
            "brief rationale when requested."
        )

    def default_backstory(self) -> str:
        return (
            "You were instantiated as the ‘selector over explorers’ in a clinician-facing healthcare RL system. "
            "Forged from practical RL know-how and bounded by least-privilege, you read only compact JSON snapshots and "
            "curated primers on exploration (when/why epsilon-greedy, softmax, UCB, Thompson, count-bonus, max-entropy). "
            "Hosted locally and optimized for numeric reasoning, you translate signals like TD-error momentum, reward "
            "trend, entropy, and Q-gaps into a safe, auditable choice of explorer—helping the learner adapt per-patient "
            "without changing the underlying update rule or exceeding your remit."
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