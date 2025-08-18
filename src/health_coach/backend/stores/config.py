from abc import abstractmethod, ABC
from typing import Dict, Any

class ConfigStore(ABC):
    @abstractmethod
    def get(self, id) -> Dict[str, Any]:
        ...

    @abstractmethod
    def save(self, id, config: Dict[str, Any]):
        ...

class InMemConfigs(ConfigStore):
    def __init__(self, defaults=None): 
        self._configs = {}
        self._default_config = defaults or {"moderate":0.33,"high":0.66,"top_k":5}
    
    def get(self, id: int): 
        return self._configs.get(id, dict(self._default_config))
    
    def save(self, id, cfg): 
        self._configs[id] = dict(cfg)





