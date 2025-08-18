from abc import abstractmethod, ABC
from typing import Dict, Any, List, Callable

QTable = List[List[float]]

class QTableStore(ABC):
    @abstractmethod
    def get(self, id: int) -> QTable: 
        ...
    
    @abstractmethod
    def save(self, id: int, table: QTable) -> None:
        ...

class InMemQTables(QTableStore):
    def __init__(self, states, actions): 
        self._qtables = {}
        self._states = states
        self._actions = actions

    def get(self, id: int) -> QTable:
        qtable = self._qtables.get(id)
        if qtable is None:
            qtable = [[0.0] * self._actions for _ in range(self._states)]
            self._qtables[id] = qtable
        return qtable  

    def save(self, id: int, table: QTable): 
        self._qtables[id] = table
