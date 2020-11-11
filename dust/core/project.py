import argparse
import os
import datetime
import logging
import sys
import toml

from dust.utils.arg_cfg_parse import Namespace, ArgCfgParser

_PROJECT = None

_PROJECT_FILE = 'project.toml'

_ARG_PARSER = ArgCfgParser()

_ARG_PARSER.add_argument(
    '--proj_dir',
    help='Directory of the project')

_ARG_PARSER.add_argument(
    '--save_cfg',
    type=bool, default=True, 
    help='Save project configuration')

def argparser():
    """ Returns the global argparser
    We maintain a single argparser for all dust modules.
    All modules call this function to get the parser and
    define arguments at the module level.
    """
    return _ARG_PARSER

class Project(object):
    """
    Usually users don't create Project object themselves.
    Instead create_project() or load_project() should be used.
    Projects are configured by args.
    """
    
    def __init__(self):
        self.time_tag = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    
    def save_project(self):
        proj_file = os.path.join(self.proj_dir, _PROJECT_FILE)
        with open(proj_file, 'w') as f:
            proj_dict = {}
            proj_dict['cfg'] = self.cfg.__dict__
            toml.dump(proj_dict, f)

def inside_project():
    return isinstance(_PROJECT, Project)

def project():
    """ Returns the current project
    One should call this after a project is created or loaded
    """
    assert isinstance(_PROJECT, Project), 'You haven\'t loaded a project.'
    return _PROJECT

def create_project(sess_name, args=None):
    """ Inits and enters a project
    """
    global _PROJECT
    assert _PROJECT is None, 'Project has been created'
    
    args, cfg = _ARG_PARSER.parse_args(args)
    _ARG_PARSER.fill_default_config(cfg)
    
    proj = Project()
    
    # Either use the current dir or the one provided by args
    proj_dir = os.path.realpath(args.proj_dir or os.getcwd())
    os.makedirs(proj_dir, exist_ok=True)
    
    proj.proj_dir = proj_dir
    proj.args = args
    proj.cfg = cfg
    proj.save_project()
    
    #proj.cfg.update(args.cfg.__dict__)
    
    #with open(proj_file, 'w') as f:
    #    toml.dump({}, f)
    
    _PROJECT = proj # One may use project() after this line
    _setup_project_logger(sess_name) # One may use logging after this line
    logging.info('Session: {}'.format(sess_name))
    logging.info('Project created')
    logging.info('Args: {}'.format(proj.args))
    logging.info('Config: {}'.format(proj.cfg))
    logging.info('Timestamp: {}'.format(proj.time_tag))
    return proj

def load_project(sess_name, args=None):
    """
    """
    global _PROJECT
    assert _PROJECT is None, 'Project has been created'
    
    args, cfg = _ARG_PARSER.parse_args(args)
    proj = Project()
    
    proj_dir = os.path.realpath(args.proj_dir or os.getcwd())
    proj.proj_dir = proj_dir
    proj.args = args
    proj.cfg = cfg
    
    proj_file = os.path.join(proj_dir, _PROJECT_FILE)
    if not os.path.isfile(proj_file):
        raise RuntimeError('{} is not found in "{}"'\
            .format(_PROJECT_FILE, proj_dir))
    
    proj_dict = toml.load(proj_file)
    proj.cfg.update(proj_dict['cfg'])
    _ARG_PARSER.fill_default_config(cfg)
    
    if args.save_cfg:
        proj.save_project()
    
    _PROJECT = proj # One may use project() after this line
    _setup_project_logger(sess_name) # One may use logging after this line
    logging.info('Session: {}'.format(sess_name))
    logging.info('Project loaded')
    logging.info('Args: {}'.format(proj.args))
    logging.info('Config: {}'.format(proj.cfg))
    logging.info('Timestamp: {}'.format(proj.time_tag))
    return proj

def _setup_project_logger(sess_name=None):
    
    time_tag = project().time_tag
    
    if sess_name:
        log_filename = 'log.{}.{}.log'.format(sess_name, time_tag)
    else:
        log_filename = 'log.{}.log'.format(time_tag)
    log_pathname = os.path.join(project().proj_dir, 'logs', log_filename)
    
    logger = logging.getLogger(None)
    logger.setLevel(logging.INFO)
    # e.g. 2020-10-08 22:26:03,509 INFO
    formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s')
    
    # Log to STDERR
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Log to file
    os.makedirs(os.path.dirname(os.path.abspath(log_pathname)), exist_ok=True)
    handler = logging.FileHandler(log_pathname)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
