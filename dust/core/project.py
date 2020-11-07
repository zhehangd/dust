import argparse
import os
import datetime
import logging
import sys
import toml

from dust.utils.argparse import Namespace

_PROJECT = None

_PROJECT_FILE = 'project.toml'

_ARG_PARSER = argparse.ArgumentParser()

_ARG_PARSER.add_argument('--proj_name', help='Name of the project')

_ARG_PARSER.add_argument('--proj_dir', help='Directory of the project')

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
    
    def __init__(self, args=None):
      
        args = _ARG_PARSER.parse_args(args)
        self.time_tag = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        self.args = args
        
        # TODO proj_name, proj_dir -> cfg
        self.proj_name = ''
        self.proj_dir = ''
        
        # TODO: If a project is loaded, updating cli args should be after
        # loading project config.
        self.cfg = Namespace()
        self.cfg.update(args.__dict__)
    
    def _parse_proj_dir(self):
        """ Determines the project dir
        The project directory is the first valid option of the followings
          * args.proj_dir
          * Directory with name args.proj_name in the current working directory
          * Current working directory
        """
        proj_dir = self.args.proj_dir or self.args.proj_name
        return os.path.realpath(proj_dir) if proj_dir else os.getcwd()
    
    def init_project(self):
        """ Initializes a new project based on the current attributes
        """
        args = self.args
        proj_dir = self._parse_proj_dir()
        proj_name = args.proj_name or os.path.basename(proj_dir)
        #if os.path.isdir(proj_dir):
        #    raise RuntimeError('{} is an existing directory'.format(proj_dir))
        os.makedirs(proj_dir, exist_ok=True)
        self.proj_dir = proj_dir
        self.proj_name = proj_name
        
    def load_project(self):
        """ Loads project data
        """
        args = self.args
        proj_dir = self._parse_proj_dir()
        
        proj_file = os.path.join(proj_dir, _PROJECT_FILE)
        if not os.path.isfile(proj_file):
            raise RuntimeError('{} is not found in "{}"'\
                .format(_PROJECT_FILE, proj_dir))
        
        proj_dict = toml.load(proj_file)
        proj_name = proj_dict['proj_name']
        self.proj_dir = proj_dir
        self.proj_name = proj_name
    
    def __str__(self):
        return 'project {}@{}'.format(
            self.proj_name, self.proj_dir,
        )
    
    def save_project(self):
        # temp
        self.cfg.update(dict(proj_name=self.proj_name))
        proj_file = os.path.join(self.proj_dir, _PROJECT_FILE)
        with open(proj_file, 'w') as f:
            toml.dump(self.cfg.__dict__, f)

def inside_project():
    return isinstance(_PROJECT, Project)

def project():
    """ Returns the current project
    One should call this after a project is created or loaded
    """
    assert isinstance(_PROJECT, Project), 'You haven\'t loaded a project.'
    return _PROJECT

def create_project(module_name, args=None):
    """ Inits and enters a project
    """
    global _PROJECT
    assert _PROJECT is None
    proj = Project(args)
    proj.init_project()
    _PROJECT = proj # One may use project() after this line
    _setup_project_logger() # One may use logging after this line
    proj.save_project()
    logging.info('Project {} created'.format(proj.proj_name))
    return proj

def load_project(module_name, args=None):
    """
    """
    global _PROJECT
    assert _PROJECT is None
    proj = Project(args)
    proj.load_project()
    _PROJECT = proj # One may use project() after this line
    _setup_project_logger(module_name) # One may use logging after this line
    # HACK: update cli args again to overwrite  
    proj.cfg.update(proj.args.__dict__)
    proj.save_project()
    logging.info('Project {} loaded'.format(proj.proj_name))
    return proj

def _setup_project_logger(module_name=None):
    
    time_tag = project().time_tag
    
    if module_name:
        log_filename = 'log.{}.{}.log'.format(module_name, time_tag)
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
