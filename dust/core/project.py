import argparse
import datetime
import logging
import os
import sys
import toml

import dust.core.init
from dust.utils import arg_and_cfg_parser

_PROJECT = None

_PROJECT_FILE = 'project.toml'

class Project(object):
    """
    Usually users don't create Project object themselves.
    Instead create_project() or load_project() should be used.
    Projects are configured by args.
    """
    
    def __init__(self, sess_name, proj_dir, timestamp, init, args):
        self.timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        self.proj_dir = os.getcwd()
        self.sess_name = sess_name
        
        
        proj_file = os.path.join(self.proj_dir, _PROJECT_FILE)
        cfg = arg_and_cfg_parser.Namespace()
        
        if not init:
            if not os.path.isfile(proj_file):
                raise RuntimeError('{} is not found in "{}"'\
                    .format(_PROJECT_FILE, self.proj_dir))
            proj_dict = toml.load(proj_file)
            cfg.__dict__.update(proj_dict['cfg'])
        
        args, cfg = dust.core.init.argparser().parse_args(args, cfg)
        self.cfg = cfg
        self.args = args
        
        logging.info('Session: {}'.format(self.sess_name))
        logging.info('Args: {}'.format(self.args))
        logging.info('Config: {}'.format(self.cfg))
        logging.info('Timestamp: {}'.format(self.timestamp))
    
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

def _create_project(sess_name, args, init):
    """ Inits and enters a project
    """
    global _PROJECT
    assert _PROJECT is None, 'Project has been created'
    
    timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    proj_dir = os.getcwd()
    
    log_file = os.path.join(proj_dir, 'logs', 'log.{}.{}.log'.format(sess_name, timestamp))
    _setup_project_logger(log_file) # One may use logging after this line
    _PROJECT = Project(sess_name=sess_name, proj_dir=proj_dir,
                   timestamp=timestamp, init=True, args=args)
    return _PROJECT

def create_project(sess_name='default', args=None):
    return _create_project(sess_name, args, True)

def load_project(sess_name, args=None):
    return _create_project(sess_name, args, False)

def _setup_project_logger(log_file=None):
    logger = logging.getLogger(None)
    logger.setLevel(logging.INFO)
    # e.g. 2020-10-08 22:26:03,509 INFO
    formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s')
    
    # Log to STDERR
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Log to file
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
