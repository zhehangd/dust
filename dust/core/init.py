import os
import sys

from typing import Callable

from dust.core.env import EnvCore, EnvAIStub, EnvDisplay
from dust.core.ai_engine import AIEngine
from dust.utils._arg_cfg_parse import ArgCfgParser

_ENV_REGISTRY = {}

_AI_ENGINE_REGISTRY = {}

_ARG_PARSER = ArgCfgParser()

class EnvRecord(object):
    """ Stub of a registered environment.
    """
    
    def __init__(self):
        self._name = ""
        self._create_env = None
        self._create_ai_stub = None
        self._create_disp = None
        self._register_args = None

def register_env(name: str,
                 create_env: Callable,
                 create_ai_stub: Callable,
                 create_disp: Callable,
                 register_args) -> None:
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
    record = EnvRecord()
    record._name = name
    record._create_env = create_env
    record._create_ai_stub = create_ai_stub
    record._create_disp = create_disp
    record._register_args = register_args
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
        if record._register_args:
            sys.stderr.write('Register env {} arguments.\n'.format(env_name))
            record._register_args()
        else:
            sys.stderr.write('Env {} does not have argument function.\n')

class AIEngineRecord(object):
    """
    
    Holds environment infomation and provide basic checking of the callback.
    """
    
    def __init__(self):
        self._name = ""
        self._create_instance = None
        self._register_args = None

def register_ai_engine(name: str, fn_create, fn_arugments) -> None:
    record = EnvRecord()
    record._name = name
    record._create_instance = fn_create
    record._register_args = fn_arugments
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
        if record._register_args:
            sys.stderr.write('Register AI engine {} arguments.\n'.format(engine_name))
            record._register_args()
        else:
            sys.stderr.write('AI engine {} does not have argument function.\n')

# TODO: moves to frames.py?

def create_training_frames(env_name: str,
                           ai_engine_name: str = None) -> None:
    
    from dust.ai_engines.prototype import PrototypeAIEngine
    from dust.core.env import EnvCore, EnvAIStub, EnvDisplay
    from dust.core import frames
    
    env_record = _ENV_REGISTRY[env_name]
    env_core = env_record._create_env()
    assert isinstance(env_core, EnvCore), type(env_core)
    env_frame = frames.EnvFrame(env_core)
    
    env_ai_stub = env_record._create_ai_stub(env_core)
    assert isinstance(env_ai_stub, EnvAIStub), type(env_ai_stub)
    
    agent = PrototypeAIEngine(env_core, env_ai_stub, False)
    ai_frame = frames.AIFrame(agent)
    
    return env_frame, ai_frame

def create_demo_frames(env_name: str,
                       ai_engine_name: str = None):
    
    from dust.core.env import EnvCore, EnvAIStub, EnvDisplay
    from dust.core.ai_engine import AIEngine
    from dust.core import frames
    
    env_record = _ENV_REGISTRY[env_name]
    env_core = env_record._create_env()
    assert isinstance(env_core, EnvCore), type(env_core)
    env_frame = frames.EnvFrame(env_core)
    
    env_ai_stub = env_record._create_ai_stub(env_core)
    assert isinstance(env_ai_stub, EnvAIStub), type(env_ai_stub)
    
    
    ai_engine_record = _AI_ENGINE_REGISTRY[ai_engine_name]
    ai_engine = ai_engine_record._create_instance(env_core, env_ai_stub, True)
    assert isinstance(ai_engine, AIEngine), type(ai_engine)
    ai_frame = frames.AIFrame(ai_engine)
    
    env_disp = env_record._create_disp(env_core, env_ai_stub)
    assert isinstance(env_disp, EnvDisplay), type(env_disp)
    disp_frame = frames.DispFrame(env_disp)
    
    return env_frame, ai_frame, disp_frame
    
def argparser() -> ArgCfgParser:
    """ Returns the global argparser
    
    We maintain a single argparser for all dust modules.
    All modules call this function to get the parser and
    define arguments at the module level.
    """
    return _ARG_PARSER

