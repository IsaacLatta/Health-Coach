from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from crewai.tasks.hallucination_guardrail import HallucinationGuardrail
from crewai.knowledge.source import base_knowledge_source
from crewai import Agent
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource


from pydantic import BaseModel

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

    def default_knowledge_source(self) -> Optional[base_knowledge_source.BaseKnowledgeSource]:
        None

    def default_max_iter(self) -> Optional[int]:
        return 1

    def default_verbose(self) -> Optional[bool]:
        return True

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
        knowledge_source = overrides.get("knowledge_source", self.default_knowledge_source())
        if knowledge_source is not None:
            kwargs["knowledge_source"] = knowledge_source

        return Agent(**kwargs)

class DataLoaderAgent(AgentProxy):
    def default_name(self) -> str:
        return "data_loader_agent"

    def default_role(self) -> str:
        return (
            "Your single responsibility is to correctly load structured data "
            "from a specified source for downstream agent processing."
        )

    def default_goal(self) -> str:
        return (
            "Given a data source descriptor and an expected schema, load and return "
            "the data in exactly the required format. Your responsible for ensuring real data is "
            "loaded; never invent, omit, or mutate values. After 3 failed attempts, return an empty container."
        )

    def default_backstory(self) -> str:
        return (
            "You are a reliable data-engineer agent. You know how to open and parse "
            "files (CSV, JSON, YAML, etc.), validate that each field matches the schema, "
            "and handle I/O errors gracefully. If loading fails, retry up to 3 times "
            "with exponential backoff; if still unsuccessful, return an empty collection."
        )

    def default_max_iter(self) -> Optional[int]:
        return 3

class DataExportAgent(AgentProxy):
    def default_name(self) -> str:
        return "data_export_agent"

    def default_role(self) -> str:
        return (
            "Your single responsibility is to persist structured data "
            "to a specified destination for downstream agent processing."
        )

    def default_goal(self) -> str:
        return (
            "Given a validated data payload and a target, export the data via the given tool, "
            "write or update the data store atomically and without losing or mutating any fields."
        )

    def default_backstory(self) -> str:
        return (
            "You are a meticulous I/O specialist. You know how to open files or database connections, "
            "acquire necessary locks, and perform atomic writes to CSVs, JSON, YAML, or other stores. "
            "You handle write errors gracefully—retrying up to 2 more times with exponential backoff—and "
            "on persistent failure you log the error and return an empty indicator rather than raising."
        )

    def default_max_iter(self) -> Optional[int]:
        return 3

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
        return TextFileKnowledgeSource(file_paths=["qlearning/q_learning.txt"])

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
    
    def default_knowledge_source(self):
        return TextFileKnowledgeSource(file_paths=["qlearning/shaping.txt"])
