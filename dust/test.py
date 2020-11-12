import importlib

env_name = 'a'
env_module = importlib.import_module('dust.envs.' + env_name)

print(env_module.create_env)
