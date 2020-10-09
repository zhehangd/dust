import os
import datetime
import logging
import sys
import toml

_PROJECT = None

_PROJECT_FILE = 'project.toml'

class Project(object):
    def __init__(self):
        self.proj_name = 'unnamed'
        self.proj_dir = "N/A"
    
    def __str__(self):
        return 'project {}@{}'.format(
            self.proj_name, self.proj_dir,
        )
    
    def save_project(self):
        global _PROJECT_FILE
        proj_file = os.path.join(self.proj_dir, _PROJECT_FILE)
        with open(proj_file, 'w') as f:
            toml.dump(dict(proj_name = self.proj_name), f)

def project():
    """ Returns the current project
    One should call this after a project is created or loaded
    """
    global _PROJECT
    assert isinstance(_PROJECT, Project)
    return _PROJECT

def create_project(proj_name=None, proj_dir=None):
    """ Inits and enters a project
    """
    global _PROJECT
    assert _PROJECT is None
    
    assert proj_name or proj_dir
    proj_name = proj_name or os.path.basename(proj_dir)
    proj_dir = proj_dir or proj_name
    if os.path.isdir(proj_dir):
        raise RuntimeError('{} is an existing directory'.format(proj_dir))
    
    proj = Project()
    proj.proj_name = proj_name
    proj.proj_dir = os.path.realpath(proj_dir)
    _PROJECT = proj # One may use project() after this line
    _setup_project_logger() # One may use logging after this line
    proj.save_project()
    logging.info('Project {} created'.format(proj_name))

def load_project(proj_name=None, proj_dir=None):
    """
    """
    global _PROJECT
    global _PROJECT_FILE
    assert _PROJECT is None
    assert proj_name or proj_dir
    
    proj_dir = proj_dir or proj_name
    proj_file = os.path.join(proj_dir, _PROJECT_FILE)
    if not os.path.isfile(proj_file):
        raise RuntimeError('{} is not found in "{}"'\
            .format(_PROJECT_FILE, proj_dir))
    
    proj_dict = toml.load(proj_file)
    stored_proj_name = proj_dict['proj_name']
    
    if proj_name and (proj_name != stored_proj_name):
        raise RuntimeError(
            'Attemped to load project {}, but the recorded name is {}'\
                .format(proj_name, stored_proj_name))
    
    proj = Project()
    proj.proj_name = stored_proj_name
    proj.proj_dir = os.path.realpath(proj_dir)
    _PROJECT = proj # One may use project() after this line
    _setup_project_logger() # One may use logging after this line
    logging.info('Project {} loaded'.format(proj_name))

def create_or_load_project(proj_name=None, proj_dir=None):
    assert proj_name or proj_dir
    proj_name = proj_name or os.path.basename(proj_dir)
    proj_dir = proj_dir or proj_name
    if os.path.isdir(proj_dir):
        load_project(proj_name=proj_name, proj_dir=proj_dir)
    else:
        create_project(proj_name=proj_name, proj_dir=proj_dir)

def _setup_project_logger():
    logtime = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    logfile = os.path.join(project().proj_dir, 'logs', '{}.log'.format(logtime))
    
    logger = logging.getLogger(None)
    logger.setLevel(logging.INFO)
    # e.g. 2020-10-08 22:26:03,509 INFO
    formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s')
    
    # Log to STDERR
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Log to file
    os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
    handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
