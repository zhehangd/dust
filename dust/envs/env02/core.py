import logging

import numpy as np

import dust.core.env

_MAP_WIDTH = 256
_MAP_HEIGHT = 256

class EnvCore(dust.core.env.EnvCore):
    
    """
    
    Attributes:
    """
    
    def __init__(self, project, state_dict: dict = None):
        super().__init__()
        self._proj = project
        self._curr_tick = 0
        
        if state_dict:
            self._curr_tick = state_dict['curr_tick']
    
    @classmethod
    def create_new_instance(cls, project):
        return cls(project)
    
    @classmethod
    def create_from_state_dict(cls, project, state_dict):
        return cls(project, state_dict)
    
    def next_tick(self):
        self._curr_tick += 1
    
    @property
    def curr_tick(self):
        return self._curr_tick
        
    def evolve(self):
        pass
    
    def update(self):
        pass
    
    def state_dict(self) -> dict:
        return dict(curr_tick=self._curr_tick)

