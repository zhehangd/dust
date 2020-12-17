import importlib

from dust import _dust
from dust.core.env import EnvCore, EnvAIStub, EnvDisplay, EnvRecord

class Env02Record(object):
    
    def import_core(self) -> type(EnvCore):
        core_module = importlib.import_module(__package__ + '.core')
        return core_module.EnvCore
    
    def import_ai_module(self, name=None) -> type(EnvAIStub):
        ai_module = importlib.import_module(__package__ + '.ai_module')
        return ai_module.AIModule
    
    def import_disp_module(self, name=None) -> type(EnvDisplay):
        disp_module = importlib.import_module(__package__ + '.disp')
        return disp_module.DispModule

_dust.register_env("env02", Env02Record())
