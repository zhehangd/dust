import os
import sys

from dust.utils import arg_and_cfg_parser

_ARG_PARSER = arg_and_cfg_parser.ArgumentAndConfigParser()

def argparser() -> arg_and_cfg_parser.ArgumentAndConfigParser:
    """ Returns the global argparser
    
    We maintain a single argparser for all dust modules.
    All modules call this function to get the parser and
    define arguments at the module level.
    """
    return _ARG_PARSER

