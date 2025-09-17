from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from crewai import Task, Agent
from crewai.tools import BaseTool

class TaskProxy(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def default_description(self) -> Optional[str]:
        ...

    @abstractmethod
    def default_tools(self) -> Optional[List[BaseTool]]:
        ...

    @abstractmethod
    def default_context(self) -> Optional[List[Task]]:
        ...

    @abstractmethod
    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def default_output_json(self) -> Optional[Type[BaseModel]]:
        ...

    @abstractmethod
    def default_expected_output(self) -> Optional[str]:
        ...

    def create(
        self,
        agent: Agent,
        *,
        tools: Optional[List[BaseTool]] = None,
        context: Optional[List[Task]] = None,
        tools_input: Optional[Dict[str, Any]] = None,
        output_json: Optional[Type[BaseModel]] = None,
        expected_output: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Task:
        kwargs: Dict[str, Any] = {
            "agent": agent,
            "name": self.name(),
        }
        
        desc = description if description is not None else self.default_description()
        if desc is not None:
            kwargs["description"] = desc
        tls = tools if tools is not None else self.default_tools()
        if tls is not None:
            kwargs["tools"] = tls
        ctx = context if context is not None else self.default_context()
        if ctx is not None:
            kwargs["context"] = ctx
        ti = tools_input if tools_input is not None else self.default_tools_input()
        if ti is not None:
            kwargs["tools_input"] = ti
        oj = output_json if output_json is not None else self.default_output_json()
        if oj is not None:
            kwargs["output_json"] = oj
        eo = expected_output if expected_output is not None else self.default_expected_output()
        if eo is not None:
            kwargs["expected_output"] = eo

        return Task(**kwargs)

class ActionResponse(BaseModel):
    select: int

class ShapeAction(TaskProxy):
    def name(self) -> str:
        return "shape_action"

    def default_description(self) -> Optional[str]:
        return (
            "You must choose exactly one exploration strategy by its index. Use this mapping:\n"
            "0 : epsilon_greedy\n"
            "1 : softmax\n"
            "2 : ucb\n"
            "3 : thompson_sampling\n"
            "4 : count_bonus\n"
            "5 : maxent\n\n"
            "Return ONLY valid JSON in the form:\n"
            "{ \"select\": <int> }\n"
            "where <int> is an integer from 0 to 5 corresponding to your chosen strategy."
        )

    def default_tools(self) -> Optional[List[BaseTool]]:
        return None

    def default_context(self) -> Optional[List[Task]]:
        return None
    
    def default_tools_input(self) -> Optional[Dict[str, Any]]:
        return None

    def default_output_json(self) -> Optional[Type[BaseModel]]:
        return ActionResponse

    def default_expected_output(self) -> Optional[str]:
        return "{select: int}"
