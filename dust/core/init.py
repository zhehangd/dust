import logging
import os
import sys

from dust.utils import arg_and_cfg_parser

_ARG_PARSER = arg_and_cfg_parser.ArgumentAndConfigParser()

_ENV_REGISTRY = {}

def argparser() -> arg_and_cfg_parser.ArgumentAndConfigParser:
    """ Returns the global argparser
    
    We maintain a single argparser for all dust modules.
    All modules call this function to get the parser and
    define arguments at the module level.
    """
    return _ARG_PARSER

def register_env(name: str, record) -> None:
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

def find_env(name: str):
    return _ENV_REGISTRY[name]

def _import_all_modules_from_package(pkg_name: str, type_name: str,
                                     dst_dict: dict) -> None:
    import importlib
    root_pkg = importlib.import_module(pkg_name)
    root_dir = os.path.dirname(root_pkg.__file__)
    for mod_name in os.listdir(root_dir):
        env_module_path = os.path.join(root_dir, mod_name)
        env_init_path = os.path.join(env_module_path, '__init__.py')
        if os.path.isdir(env_module_path) and os.path.isfile(env_init_path):
            logging.warning('Importing {} {} ...\n'.format(type_name, mod_name))
            num_records = len(dst_dict)
            # __init__ module should do the registration
            module = importlib.import_module('{}.{}'.format(pkg_name, mod_name))
            if num_records == len(dst_dict):
                logging.warning('Module {} didn\'t register any {}.\n'.format(mod_name, type_name))
                logging.warning(dst_dict)

def register_all_envs() -> None:
    """ Registers environments from all known sources
    This function should be called once before creating or loading a project,
    if one expects to interact with any environment.
    """
    _import_all_modules_from_package('dust.envs', 'env', _ENV_REGISTRY)
