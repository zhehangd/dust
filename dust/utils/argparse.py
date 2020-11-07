import argparse
#import toml


class ActionBool(argparse.Action):
    """ Supports optional boolean argument
    This is a copy of the CPython implementation of \
    argparse.BooleanOptionalAction in Python 3.9+
    Use it with default arg value 
    """
    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith('--'):
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)

        if help is not None and default is not None:
            help += f" (default: {default})"

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith('--no-'))

    def format_usage(self):
        return ' | '.join(self.option_strings)
      
      
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

