import importlib

path = __package__

print(path)

def create_env():
    core_module = importlib.import_module(__package__ + '.core')
    return core_module.ProtoEnv()

def create_disp():
    return None
