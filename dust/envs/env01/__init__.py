import importlib

from dust import _dust

def _create_env(state_dict: dict = None):
    core_module = importlib.import_module(__package__ + '.core')
    return core_module.Env01Core(state_dict)

def _create_ai_stub(env, state_dict: dict = None):
    ai_stub_module = importlib.import_module(__package__ + '.ai_stub')
    return ai_stub_module.Env01Stub(env, state_dict)

def _create_disp(env, ai, state_dict: dict = None):
    disp_module = importlib.import_module(__package__ + '.disp')
    return disp_module.Disp(env, ai, state_dict)

def _register_args():
    pass

_dust.register_env("env01",
    _create_env, _create_ai_stub,
    _create_disp, _register_args)
