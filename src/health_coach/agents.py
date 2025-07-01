from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from crewai import Agent
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

    @abstractmethod
    def default_max_iter(self) -> Optional[int]:
        ...

    @abstractmethod
    def default_verbose(self) -> Optional[bool]:
        ...

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

    def default_verbose(self) -> Optional[bool]:
        return True

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

    def default_verbose(self) -> Optional[bool]:
        return True

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

    def default_max_iter(self) -> Optional[int]:
        return 1

    def default_verbose(self) -> Optional[bool]:
        return True


class RewardShapingAgent(AgentProxy):
    def default_name(self) -> str:
        return "reward_shaping_agent"

    def default_role(self) -> str:
        return (
            "Your role is to be the expert reward shaper in the reinforcement learning loop. "
            "You determine the quality of transitions and apply those insights to improve learning."
        )

    def default_goal(self) -> str:
        return (
            "Given a previous state and a current state, compute the scalar reward, "
            "then update the underlying RL model to incorporate this reward."
        )

    def default_backstory(self) -> str:
        return (
            "You are the sentient instantiation of a reward function, born to evaluate transitions. "
            "With profound intuition about desirable and undesirable outcomes, you wield the necessary "
            "tools to shape learning."
        )

    def default_max_iter(self) -> Optional[int]:
        return 1

    def default_verbose(self) -> Optional[bool]:
        return True