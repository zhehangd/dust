import importlib

from dust import _dust
from dust.core.env import EnvCore, EnvAIStub, EnvDisplay, EnvRecord

class Env01Record(object):
    
    def import_core(self) -> type(EnvCore):
        core_module = importlib.import_module(__package__ + '.core')
        return core_module.Env01Core
    
    def import_ai_module(self, name=None) -> type(EnvAIStub):
        ai_stub_module = importlib.import_module(__package__ + '.ai_stub')
        return ai_stub_module.Env01Stub
    
    def import_disp_module(self, name=None) -> type(EnvDisplay):
        disp_module = importlib.import_module(__package__ + '.disp')
        return disp_module.Disp

_dust.register_env("env01", Env01Record())
