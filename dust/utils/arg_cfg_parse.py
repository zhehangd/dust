
""" Command-line parsing utilities

This module provides several utilities that can be used with the argparse
built-in module.

There are several features we want to made:
    * Organizing all options into two groups:
        1. Regular options apply only to the current execution
        2. Cached options that can be saved in the project config file
    * Supporting the use of --foo/--no-foo bool options.
        Argparse has a 'store_true' action, but the result is either
        True or False. We want to allow an "unspecified" state to
        indicate the program should either use the default value
        or the cached one. Since Python 3.9 there is a BooleanOptionalAction
        action that does exactly this, but we would like to support
        older versions as well.


Usage Examples

    parser.add_argument('--foo', action=_dust.__ArgumentAction, ...)
    parser.add_argument('--bar, action=_dust.__ConfigAction, ...)

"""

import argparse

class Namespace(object):
  """ A simple object subclass that provides attribute access to its namespace
  This class is very like types.SimpleNamespace. One major difference is that
  this class ignores any item with a None value.
  """
  
  def __init__(self):
      pass
  
  def update(self, d):
      """ Adds items in a dict to the namespace
      Existing items are overwritten. An item is ignored if its value is None
      """
      assert isinstance(d, dict)
      filtered_d = {k: v for k, v in d.items() if v is not None}
      self.__dict__.update(filtered_d)
      
  def __repr__(self):
      items = (f"{k}={v!r}" for k, v in self.__dict__.items())
      return "{}({})".format(type(self).__name__, ", ".join(items))

class _Action(argparse.Action):
    
    _GROUP = None

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):

        self.boolean = type == bool

        if self.boolean:
            type = None
            nargs = 0
            assert const is None
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith('--'):
                    option_string = '--no-' + option_string[2:]
                    _option_strings.append(option_string)
        else:
            if nargs == 0:
                raise ValueError('nargs for store actions must be != 0; if you '
                                 'have nothing to store, actions such as store '
                                 'true or store const may be more appropriate')
            if const is not None and nargs != OPTIONAL:
                raise ValueError('nargs must be %r to supply const' % OPTIONAL)
            _option_strings = option_strings
            
        
        super(_Action, self).__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=nargs,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)


    def __call__(self, parser, namespace, values, option_string=None):
        if self._GROUP:
            if hasattr(namespace, self._GROUP):
                namespace = getattr(namespace, self._GROUP)
            else:
                setattr(namespace, self._GROUP, Namespace())
                namespace = getattr(namespace, self._GROUP)
        
        if self.boolean:
            if option_string in self.option_strings:
                val = not option_string.startswith('--no-')
                setattr(namespace, self.dest, val)
        else:
            setattr(namespace, self.dest, values)
        
    def format_usage(self):
        if self.boolean:
            return ' | '.join(self.option_strings)
        else:
            return self.option_strings[0]

class _ArgumentAction(_Action):
    _GROUP = "args"

class _ConfigAction(_Action):
    _GROUP = "cfg"

class ArgCfgParser(object):
    
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._cfg_default = {}
        self._arg_default = {}
    
    def add_configuration(self, *args, **kwargs):
        assert args
        assert all(name.startswith('-') for name in args)
        name = args[0].lstrip('-')
        
        if 'default' in kwargs:
            self._cfg_default[name] = kwargs['default']
            del kwargs['default']
        else:
            self._cfg_default[name] = None
        
        kwargs['action'] = _ConfigAction
        self._parser.add_argument(*args, **kwargs)
    
    def add_argument(self, *args, **kwargs):
        assert args
        name = args[0].lstrip('-')
        
        if 'default' in kwargs:
            self._arg_default[name] = kwargs['default']
            del kwargs['default']
        else:
            self._arg_default[name] = None
        
        kwargs['action'] = _ArgumentAction
        self._parser.add_argument(*args, **kwargs)
        
    def parse_args(self, args=None):
        _groups = Namespace()
        _groups.cfg = Namespace()
        _groups.args = Namespace()
        
        self._parser.parse_args(args, _groups)
        
        for name, val in self._arg_default.items():
            if not hasattr(_groups.args, name):
                setattr(_groups.args, name, val)
        
        return _groups.args, _groups.cfg
    
    def fill_default_config(self, namespace):
        """ Adds absent entries with default values
        """
        for name, val in self._cfg_default.items():
            if not hasattr(namespace, name):
                setattr(namespace, name, val)

