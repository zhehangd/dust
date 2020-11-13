from dust.utils import _arg_cfg_parse

_ENV_REGISTRY = {}

_AIENGINE_REGISTRY = {}

_ARG_PARSER = _arg_cfg_parse.ArgCfgParser()

class _EnvRecord(object):
    def __init__(self):
        self.name = ""
        self.create_env = None
        self.create_ai_stub = None
        self.create_disp = None
        self.register_args = None

def register_env(name, create_env, create_ai_stub,
                 create_disp, register_args):
    record = _EnvRecord()
    record.name = name
    record.create_env = create_env
    record.create_ai_stub = create_ai_stub
    record.create_disp = create_disp
    record.register_args = register_args
    if hasattr(_ENV_REGISTRY, name):
        raise RuntimeError('"{}" is a registered env.'.format(name))
    _ENV_REGISTRY[name] = record

def argparser():
    """ Returns the global argparser
    We maintain a single argparser for all dust modules.
    All modules call this function to get the parser and
    define arguments at the module level.
    """
    return _ARG_PARSER

