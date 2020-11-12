import importlib

def create_env():
    core_module = importlib.import_module(__package__ + '.core')
    return core_module.Env()

def create_ai_stub(env):
    ai_stub_module = importlib.import_module(__package__ + '.ai_stub')
    return ai_stub_module.Env01Stub(env)
    

def create_disp(env, ai):
    disp_module = importlib.import_module(__package__ + '.disp')
    return disp_module.Disp(env, ai)
