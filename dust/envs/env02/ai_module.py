import logging

import numpy as np

from dust.core.env import EnvAIStub
from dust.utils import np_utils
from dust.core import progress_log

from dust.ai_engines import prototype

from dust import _dust

BRAIN_NAME = 'brain01'
TERM_NAME = 'term01'

_ARGPARSER = _dust.argparser()

#_ARGPARSER.add_argument('--freeze', action='store_true',
#    help='')

class AIModule(EnvAIStub):
    
    def __init__(self, project, env, state_dict: dict = None):
        self._proj = project
        if state_dict is not None:
            engine = prototype.Engine.create_from_state_dict(state_dict['engine'])
        else:
            brain_def = prototype.BrainDef(25, 4, (16, 16))
            engine = prototype.Engine.create_new_instance()
            engine.add_brain(BRAIN_NAME, brain_def)
            engine.add_terminal(TERM_NAME, BRAIN_NAME)
        
        self.env = env
        self.engine = engine
        self.progress = progress_log.ProgressLog(project)

        
    def state_dict(self) -> dict:
        return {'engine': self.engine.state_dict()}

    @classmethod
    def create_new_instance(cls, project, env_core) -> 'EnvAIStub':
        return cls(project, env_core)
    
    @classmethod
    def create_from_state_dict(cls, project, env_core, state_dict) -> 'EnvAIStub':
        return cls(project, env_core, state_dict)
    
    def perceive_and_act(self) -> None:
        """ Perceives environment state and takes actions
        """
        pass

    def update(self) -> None:
        pass
    
