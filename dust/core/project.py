import datetime
import logging
import os
import re
import sys
import tempfile

import toml

import dust.core.init
from dust.utils.arg_and_cfg_parser import Namespace

_PROJECT_FILE = 'project.toml'

# TODO: Maybe global project is a bad idea. Just let all objects take a
#       project object. Only thoese who do so have the right to define
#       args/cfg and make logging. This makes sense.
#
# TODO: Clarify the exact usages of the context manager
#         Guaranteed release?

class Project(object):
    """
    
    A Project instance holds the context of a Dust session.
    It provides the access to the arguments, configuration,
    logging, and other information to all Dust classes and
    methods.
    
    Typically, users create a project by calling create_project,
    load_project, or create_temporary_project, instead of constructing
    an object directly.
    
    Attributes:
    
        args (Namespace): Arguments
        
        cfg (Namespace): Project configuration
        
        log_filename (str): Filename of the log file.
        
        proj_dir (str): Filename of the project directory.
        
        sess_name (str): Name of the session.
        
        timestamp (str): Session timestamp.
        
        log (logging.Logger): Logger of the session
    
    Examples:
        
        Creating a temporary project
        
        with create_temporary_project() as proj:
            proj.parse_args()
            proj.log_proj_info()
            
    
    """
    
    def __init__(self, load_proj: bool, **kwargs):
        self._timestamp = kwargs.pop('timestamp', self._get_timestamp())
        self._proj_dir = kwargs.pop('proj_dir', os.getcwd())
        self._sess_name = kwargs.pop('sess_name', 'default')
        self._init_logger()
        
        proj_file = os.path.join(self.proj_dir, _PROJECT_FILE)
        cfg = Namespace()
        args = Namespace()
        if load_proj:
            if not os.path.isfile(proj_file):
                raise RuntimeError('{} is not found in "{}"'\
                    .format(_PROJECT_FILE, self.proj_dir))
            proj_dict = toml.load(proj_file)
            cfg.__dict__.update(proj_dict['cfg'])
        self.cfg = cfg
        self.args = args
        
        self._temp_dir = kwargs.pop('temp_dir', None)
        if self._temp_dir is not None:
            assert isinstance(self._temp_dir, tempfile.TemporaryDirectory)
        
        assert len(kwargs) == 0, \
            'Unknown arguments {}'.format(', '.join(kwargs.keys()))
    
    def __del__(self):
        self.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, trace):
        self.release()
    
    def release(self):
        """ Releases all external resources the project holds
        
        The logger handlers the project added are removed. If a temporary
        directory is given, it gets cleaned up. 
        
        One should not use the project again after this method is called.
        This method is called automatically when the object gets destructed.
        """
        logger = logging.getLogger(None)
        for handler in self._log_handlers:
            logger.removeHandler(handler)
        self._log_handlers = []
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None
    
    def renew_timestamp(self, timestamp=None):
        """ Update the session timestamp
        
        'timestamp' can be str or None (default).
        If it is None the timestamp is updated according to the current time.
        If it is a str then it is used as the new timestamp.
        
        The timestamp must have the form of 'YYYY-MM-DD-HH-MM-SS' or a 
        ValueError is raised.
        
        """
        
        if timestamp is not None:
            pattern = '[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}'
            assert re.fullmatch(pattern, timestamp), \
                'Timestamp is not a valid string.'
            self._timestamp = timestamp 
        else:
            self._timestamp = self._get_timestamp()
    
    def parse_args(self, args=None, allow_unknown=False):
        """ Parse arguments and configuration
        
        Parameters:
        
            args (list | None): A optional list of strings to replace the
                command-line arguments
            
            allow_unknown (bool): Ignore unknown arguments instead of
                raising an Exception. 
        
        """
        parser = dust.core.init.argparser()
        args, cfg = parser.parse_args(args, self.cfg, allow_unknown=allow_unknown)
        self.cfg = cfg
        self.args = args
    
    def log_proj_info(self):
        """ Logs the content of the project
        """
        logging.info('Session: {}'.format(self.sess_name))
        logging.info('Args: {}'.format(self.args))
        logging.info('Config: {}'.format(self.cfg))
        logging.info('Timestamp: {}'.format(self.timestamp))
    
    def save_project(self):
        """ Writes the project file to the disk.
        """
        proj_file = os.path.join(self.proj_dir, _PROJECT_FILE)
        with open(proj_file, 'w') as f:
            proj_dict = {}
            proj_dict['cfg'] = self.cfg.__dict__
            toml.dump(proj_dict, f)
    
    def relpath(self, path):
        """
        """
        return os.path.relpath(path, self._proj_dir)
    
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
    
    def _init_logger(self):
        logger = logging.getLogger(None)
        logger.setLevel(logging.INFO) # TODO attach to handlers?
        
        self.log = logger
        self._log_handlers = []
        
        # e.g. 2020-10-08 22:26:03,509 INFO
        formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s')
        
        # Log to STDERR
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self._log_handlers.append(handler)
        
        # Log to file
        log_basename = 'log.{}.{}.log'.format(self.sess_name, self.timestamp)
        log_filename = os.path.join(self.proj_dir, 'logs', log_basename)
        self._log_filename = log_filename
        
        os.makedirs(os.path.dirname(os.path.abspath(log_filename)), exist_ok=True)
        handler = logging.FileHandler(log_filename)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self._log_handlers.append(handler)
        #for filter in logger.filters[:]:
        #    logger.removeFilter(filter)
    
    def _get_timestamp(self):
        return datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

def create_project(**kwargs) -> Project:
    return Project(False, **kwargs)

def load_project(**kwargs) -> Project:
    return Project(True, **kwargs)

def create_temporary_project(**kwargs) -> Project:
    assert 'proj_dir' not in kwargs
    assert 'temp_dir' not in kwargs
    temp_dir_obj = tempfile.TemporaryDirectory()
    proj_dir = temp_dir_obj.name
    kwargs['proj_dir'] = proj_dir
    kwargs['temp_dir'] = temp_dir_obj
    proj = Project(False, **kwargs)
    assert proj.proj_dir == proj_dir, '{} vs. {}'.format(proj.proj_dir, proj_dir)
    # Upon the destruction of the project, this object is also destructed
    # as well as the temporary directory.
    proj._temp_dir_obj = temp_dir_obj 
    return proj

