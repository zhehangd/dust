import argparse

Namespace = argparse.Namespace

class ArgumentAndConfigParser(object):
    
    def __init__(self, **kwargs):
        self._argparser = argparse.ArgumentParser(**kwargs)
        self._prefix_chars = kwargs['prefix_chars'] if hasattr(kwargs, 'prefix_chars') else '-'
        self._arg_dests = set()
        
    def add_configuration(self, *args, **kwargs):
        self._argparser.add_argument(*args, **kwargs)
    
    def add_argument(self, *args, **kwargs):
        if not args or len(args) == 1 and args[0][0] not in self._prefix_chars:
            dest = args[0]
        else:
            dest = self._get_optional_dest(*args, **kwargs)
        self._arg_dests.add(dest)
        self._argparser.add_argument(*args, **kwargs)
        
    def parse_args(self, args=None, cfg_namespace=None):
        if cfg_namespace is None:
            cfg_namespace = Namespace()
        common_keys = cfg_namespace.__dict__.keys() & self._arg_dests
        assert len(common_keys) == 0, ''\
            'Provided config namespace conflicts with '\
            'the defined arguments {}'.format(common_keys)
        self._argparser.parse_args(args, cfg_namespace)
        arg_namespace = argparse.Namespace()
        for arg_name in self._arg_dests:
            setattr(arg_namespace, arg_name, getattr(cfg_namespace, arg_name))
            delattr(cfg_namespace, arg_name)
        return arg_namespace, cfg_namespace
    
    def _get_optional_dest(self, *args, **kwargs):
        dest = kwargs.pop('dest', None)
        if dest is not None:
            return dest
        assert len(args) > 0
        for name in args:
            if len(name) > 1 and name[1] in self._prefix_chars:
                dest = name
                break
        if dest is None:
            dest = args[0]
        dest = dest.lstrip(self._prefix_chars)
        dest = dest.replace('-', '_')
        return dest