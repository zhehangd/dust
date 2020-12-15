import argparse
import datetime
import logging
import os
import sys
import tempfile

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
    
    def __init__(self, load_proj: bool, **kwargs):
        global _PROJECT
        assert _PROJECT is None, 'Project has been created'
        _PROJECT = self
        
        self._timestamp = kwargs.get('timestamp',
            datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
        self._proj_dir = kwargs.get('proj_dir', os.getcwd())
        self._sess_name = kwargs.get('sess_name', 'default')
        self._init_logger()
        
        proj_file = os.path.join(self.proj_dir, _PROJECT_FILE)
        cfg = arg_and_cfg_parser.Namespace()
        args = arg_and_cfg_parser.Namespace()
        if load_proj:
            if not os.path.isfile(proj_file):
                raise RuntimeError('{} is not found in "{}"'\
                    .format(_PROJECT_FILE, self.proj_dir))
            proj_dict = toml.load(proj_file)
            cfg.__dict__.update(proj_dict['cfg'])
        self.cfg = cfg
        self.args = args
        self._global = True
    
    def release(self):
        """ Detaches the project from the global project position
        
        This removes the project from the global project variable,
        and reset the logger it maintains, making the project a
        normal object without global effect.
        
        The only scenario this method should be called is when you
        want to switch projects for some special reasons (like testing).
        
        You may call this method only when this project is the global one.
        
        """
        assert self._global
        global _PROJECT
        assert _PROJECT == self
        _PROJECT = None
        self._reset_logger()
        self._global = False
    
    @property
    def log_filename(self):
        return self._log_filename
    
    @property
    def proj_dir(self):
        return self._proj_dir
    
    @property
    def sess_name(self):
        return self._sess_name
    
    @property
    def timestamp(self):
        return self._timestamp
    
    def parse_args(self, args=None, allow_unknown=False):
        parser = dust.core.init.argparser()
        args, cfg = parser.parse_args(args, self.cfg, allow_unknown=allow_unknown)
        self.cfg = cfg
        self.args = args
    
    def log_proj_info(self):
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
    
    def _init_logger(self):
        self._reset_logger()
        logger = logging.getLogger(None)
        logger.setLevel(logging.INFO)
        
        # e.g. 2020-10-08 22:26:03,509 INFO
        formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s')
        
        # Log to STDERR
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Log to file
        log_basename = 'log.{}.{}.log'.format(self.sess_name, self.timestamp)
        log_filename = os.path.join(self.proj_dir, 'logs', log_basename)
        self._log_filename = log_filename
        
        os.makedirs(os.path.dirname(os.path.abspath(log_filename)), exist_ok=True)
        handler = logging.FileHandler(log_filename)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def _reset_logger(self):
        assert logging
        logger = logging.getLogger(None)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        for filter in logger.filters[:]:
            logger.removeFilter(filter)

def inside_project():
    return isinstance(_PROJECT, Project)

def project():
    """ Returns the current project
    One should call this after a project is created or loaded
    """
    assert isinstance(_PROJECT, Project), 'You haven\'t loaded a project.'
    return _PROJECT

def _create_project(sess_name, proj_dir, init) -> Project:
    """ Inits and enters a project
    """
    
    _PROJECT = Project(sess_name=sess_name, proj_dir=proj_dir,
                       timestamp=timestamp, init=init)
    return _PROJECT

def create_project(**kwargs) -> Project:
    return Project(False, **kwargs)

def load_project(**kwargs) -> Project:
    return Project(True, **kwargs)

def create_temporary_project(**kwargs) -> Project:
    temp_dir_obj = tempfile.TemporaryDirectory()
    proj_dir = temp_dir_obj.name
    kwargs['proj_dir'] = proj_dir
    proj = Project(False, **kwargs)
    assert proj.proj_dir == proj_dir, '{} vs. {}'.format(proj.proj_dir, proj_dir)
    # Upon the destruction of the project, this object is also destructed
    # as well as the temporary directory.
    proj._temp_dir_obj = temp_dir_obj 
    return proj

def close_project():
    global _PROJECT
    assert isinstance(_PROJECT, Project)
    proj, _PROJECT = _PROJECT, None
    del proj

def detach_global_project():
    if _PROJECT:
        _PROJECT.release()
