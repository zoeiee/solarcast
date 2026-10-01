import yaml
import torch
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config = config_dict or {}

    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        if not os.path.exists(filepath):
            return cls({})
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict or {})

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        curr = self._config
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr

    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
