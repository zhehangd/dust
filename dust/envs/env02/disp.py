import random
import time

import numpy as np 

from dust.utils import rendering 
from dust.core.env import EnvDisplay

class DispModule(EnvDisplay):
    
    def __init__(self, project, env_core, env_ai_stub, state_dict: dict = None):
        self._core = env_core
        self._ai = env_ai_stub
        self._proj = project
    
    @classmethod
    def create_new_instance(cls, project, env_core, ai_stub):
        return cls(project, env_core, ai_stub)
    
    @classmethod
    def create_from_state_dict(cls, project, env_core, ai_stub, state_dict):
        return cls(project, env_core, ai_stub, state_dict)
    
    def init(self):
        pass
    
    def render(self):
        pass
        
    def close(self):
        pass
