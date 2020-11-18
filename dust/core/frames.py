import logging
import os
import pickle

from dust.core.init import _ENV_REGISTRY, _AI_ENGINE_REGISTRY
from dust.core.env import EnvCore, EnvAIStub, EnvDisplay
from dust.core.ai_engine import AIEngine
from dust.utils import state_dict

class DustFrame(object):
    """
    
    Attributes:
        env: ...
        ai: ...
        disp: ...
        env_name (str):
        ai_engine_name (str):
    
    """
    
    def state_dict(self) -> dict:
        sd = {}
        sd['version'] = 'dev'
        sd['env_name'] = self.env_name
        sd['ai_engine_name'] = self.ai_engine_name
        sd['env_core'] = self._env_core.state_dict()
        sd['env_ai_stub'] = self._env_ai_stub.state_dict()
        sd['ai_engine'] = self._ai_engine.state_dict()
        
        return sd
    
    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.state_dict(), f)
    
    @staticmethod
    def _create_frames(env_name: str, ai_engine_name: str, is_train: bool, state_dict: dict):
        """
        
        env_name and ai_engine_name overwrites the items in state_dict
        
        """
        env_record = _ENV_REGISTRY[env_name]
        
        env_core_sd = state_dict['env_core'] if state_dict else None
        env_core = env_record._create_env(
            state_dict=env_core_sd)
        assert isinstance(env_core, EnvCore), type(env_core)
        env_frame = EnvFrame(env_core)
        
        env_ai_stub_sd = state_dict['env_ai_stub'] if state_dict else None
        env_ai_stub = env_record._create_ai_stub(
            env_core, state_dict=env_ai_stub_sd)
        assert isinstance(env_ai_stub, EnvAIStub), type(env_ai_stub)
        
        ai_engine_record = _AI_ENGINE_REGISTRY[ai_engine_name]
        
        ai_engine_sd = state_dict['ai_engine'] if state_dict else None
        
        ai_engine_class = ai_engine_record._import_class()
        if state_dict:
            ai_engine = ai_engine_class.create_from_state_dict(env_ai_stub, state_dict, freeze=not is_train)
        else:
            ai_engine = ai_engine_class.create_new_instance(env_ai_stub, freeze=not is_train)
        assert isinstance(ai_engine, AIEngine), type(ai_engine)
        ai_frame = AIFrame(ai_engine)
        
        if not is_train:
            env_disp = env_record._create_disp(env_core, env_ai_stub)
            assert isinstance(env_disp, EnvDisplay), type(env_disp)
            disp_frame = DispFrame(env_disp)
        
        f = DustFrame()
        f.env_name = env_name
        f.ai_engine_name = ai_engine_name
        
        f.env = env_frame
        f.ai = ai_frame
        
        f._env_core = env_core
        f._env_ai_stub = env_ai_stub
        f._ai_engine = ai_engine
        
        if not is_train:
            f.disp = disp_frame
            f._env_disp = env_disp
        
        return f
    
    @staticmethod
    def create_training_frames(env_name: str, ai_engine_name: str):
        """ Creates a set of frames for training
        
        This function instantiates an environment and an AI engine,
        and loads them into an environment frame and an AI frame.
        The frames provides the interface to conduct the simulation
        in training mode.
        
        Args:
            env_name (str): Name of the environment
            ai_engine_name (str): Name of the AI Engine
        
        Returns:
            frames (tuple): An environment frame and an AI frame
        """
        return DustFrame._create_frames(env_name, ai_engine_name, True, None)
    

    @staticmethod
    def create_demo_frames(env_name: str, ai_engine_name: str):
        """ Creates a set of frames for training
        
        This function instantiates an environment and an AI engine,
        and loads them into an environment frame, an AI frame,
        and a display frame. The frames provides the interface to
        conduct the simulation in demo mode.
        
        Args:
            env_name (str): Name of the environment
            ai_engine_name (str): Name of the AI Engine
        
        Returns:
            frames (tuple): An environment frame, an AI frame, and a display frame
        """
        return DustFrame._create_frames(env_name, ai_engine_name, False, state_dict)
    
    @staticmethod
    def create_training_frames_from_save(filename):
        with open(filename, 'rb') as f:
            state_dict = pickle.loads(f.read())
        assert isinstance(state_dict, dict)
        assert state_dict['version'] == 'dev'
        env_name = state_dict['env_name']
        ai_engine_name = state_dict['ai_engine_name']
        return DustFrame._create_frames(env_name, ai_engine_name, True, state_dict)

    @staticmethod
    def create_demo_frames_from_save(filename):
        with open(filename, 'rb') as f:
            state_dict = pickle.loads(f.read())
        assert isinstance(state_dict, dict)
        assert state_dict['version'] == 'dev'
        env_name = state_dict['env_name']
        ai_engine_name = state_dict['ai_engine_name']
        return DustFrame._create_frames(env_name, ai_engine_name, False, state_dict)

class EnvFrame(object):
    """ Interface to interact with an environment
    
    """
    
    def __init__(self, env_core: EnvCore):
        self.env_core = env_core
    
    def new_simulation(self) -> None:
        """ Creates a new simulation
        """
        logging.info('Creating a new simulation')
        self.env_core.new_simulation()
    
    def next_tick(self):
        self.env_core.next_tick()
        
    def curr_tick(self) -> int:
        return self.env_core.curr_tick()

    def evolve(self) -> None:
        """ Executes one-tick simulation
        """
        self.env_core.evolve()
    
    def update(self) -> None:
        self.env_core.update()

class AIFrame(object):
    """ Interface to interact with an AI engine
    """
    
    def __init__(self, ai_engine: AIEngine):
        self.ai_engine = ai_engine
    
    def perceive_and_act(self):
        """ Perceives the environment and takes action
        """
        self.ai_engine.perceive_and_act()

    def update(self):
        """ Receives environment feedback and updates AI
        """
        self.ai_engine.update()

class DispFrame(object):
    """ Interface to display system
    """
    def __init__(self, env_disp: EnvDisplay):
        self.env_disp = env_disp
    
    def init(self) -> None:
        """ Prepares for display
        """
        self.env_disp.init()
    
    def render(self) -> None:
        """ Renders the status of the environment and AI engine
        """
        self.env_disp.render()
