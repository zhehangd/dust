import importlib

from dust import _dust

def _create_env():
    core_module = importlib.import_module(__package__ + '.core')
    return core_module.Env()

def _create_ai_stub(env):
    ai_stub_module = importlib.import_module(__package__ + '.ai_stub')
    return ai_stub_module.Env00Stub(env)

def _create_disp(env, ai):
    disp_module = importlib.import_module(__package__ + '.disp')
    return disp_module.Disp(env, ai)

def _register_args():
    pass

_dust.register_env("env00",
    _create_env, _create_ai_stub,
    _create_disp, _register_args)
