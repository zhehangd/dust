import importlib

from dust import _dust

def _create_instance(*args, **kwargs):
    core_module = importlib.import_module(__package__ + '.prototype')
    return core_module.PrototypeAIEngine(*args, **kwargs)

def _register_args():
    pass

_dust.register_ai_engine("prototype",
    _create_instance, _register_args)
 
