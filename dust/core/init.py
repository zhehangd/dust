import os
import sys

from typing import Callable

from dust.core.env import EnvCore, EnvAIStub, EnvDisplay, EnvRecord
from dust.core.ai_engine import AIEngine, AIEngineRecord
from dust.utils._arg_cfg_parse import ArgCfgParser

_ENV_REGISTRY = {}

_AI_ENGINE_REGISTRY = {}

_ARG_PARSER = ArgCfgParser()

def register_env(name: str, record: EnvRecord) -> None:
    """ Registers an environment.
    
    Registering an environment requires a name and four functions.
    A name to uniquely identify the registered environment.
    Three functions to create an environment core, an AI stub,
    and a display objects. The last function registers arugments
    for the environment.
    
    Args:
        name (str): Name of the environment.
        create_env (Callable[[], EnvCore]): X
        create_ai_stub (Callable[[EnvCore], EnvAIStub]): X
        create_disp (Callable[[EnvCore, EnvAIStub], EnvDisplay]): X
    
    """
    if hasattr(_ENV_REGISTRY, name):
        raise RuntimeError('"{}" is a registered env.'.format(name))
    _ENV_REGISTRY[name] = record

def _import_all_modules_from_package(pkg_name: str, type_name: str,
                                     dst_dict: dict) -> None:
    import importlib
    root_pkg = importlib.import_module(pkg_name)
    root_dir = os.path.dirname(root_pkg.__file__)
    for mod_name in os.listdir(root_dir):
        env_module_path = os.path.join(root_dir, mod_name)
        env_init_path = os.path.join(env_module_path, '__init__.py')
        if os.path.isdir(env_module_path) and os.path.isfile(env_init_path):
            sys.stderr.write('Importing {} {} ...\n'.format(type_name, mod_name))
            num_records = len(dst_dict)
            # __init__ module should do the registration
            module = importlib.import_module('{}.{}'.format(pkg_name, mod_name))
            if num_records == len(dst_dict):
                sys.stderr.write('Module {} didn\'t register any {}.\n'.format(mod_name, type_name))

def register_all_envs() -> None:
    """ Registers environments from all known sources
    This function should be called once before creating or loading a project,
    if one expects to interact with any environment.
    """
    _import_all_modules_from_package('dust.envs', 'env', _ENV_REGISTRY)

def register_all_env_arguments() -> None:
    """ Iterates all registered envs and registers their arguments
    This function should be called once before creating or loading a project,
    after all required envs are registered.
    """
    for env_name, record in _ENV_REGISTRY.items():
        sys.stderr.write('Register env {} arguments.\n'.format(env_name))
        record.register_args()

def register_ai_engine(name: str, record: AIEngineRecord) -> None:
    if hasattr(_AI_ENGINE_REGISTRY, name):
        raise RuntimeError('"{}" is a registered AI engine.'.format(name))
    _AI_ENGINE_REGISTRY[name] = record

def register_all_ai_engines() -> None:
    _import_all_modules_from_package('dust.ai_engines', 'ai engine',
                                     _AI_ENGINE_REGISTRY)

def register_all_ai_engine_arguments() -> None:
    """ Iterates all registered envs and registers their arguments
    This function should be called once before creating or loading a project,
    after all required envs are registered.
    """
    for engine_name, record in _AI_ENGINE_REGISTRY.items():
        sys.stderr.write('Register AI engine {} arguments.\n'.format(engine_name))
        record.register_args()

def argparser() -> ArgCfgParser:
    """ Returns the global argparser
    
    We maintain a single argparser for all dust modules.
    All modules call this function to get the parser and
    define arguments at the module level.
    """
    return _ARG_PARSER

