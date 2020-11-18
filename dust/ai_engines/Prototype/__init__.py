import importlib

from dust import _dust

def _import_class():
    core_module = importlib.import_module(__package__ + '.prototype')
    return core_module.PrototypeAIEngine

def _register_args():
    pass

_dust.register_ai_engine("prototype",
    _import_class, _register_args)
 
