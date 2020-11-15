import logging

from dust.core.init import _ENV_REGISTRY, _AI_ENGINE_REGISTRY
from dust.core.env import EnvCore, EnvAIStub, EnvDisplay
from dust.core.ai_engine import AIEngine

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
        sd = dict()
        sd['verion'] = 'dev'
        sd['env_frame'] = self.env.state_dict()
        sd['ai_frame'] = self.ai.state_dict()
        sd['env_name'] = self.env_name
        sd['ai_engine_name'] = self.ai_engine_name
        return sd
    
    def load_state_dict(self, sd: dict) -> None:
        assert sd['version'] == 'dev'
    
    @staticmethod
    def create_training_frames(env_name: str, ai_engine_name: str) -> tuple:
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
        env_record = _ENV_REGISTRY[env_name]
        env_core = env_record._create_env()
        assert isinstance(env_core, EnvCore), type(env_core)
        env_frame = EnvFrame(env_core)
        
        env_ai_stub = env_record._create_ai_stub(env_core)
        assert isinstance(env_ai_stub, EnvAIStub), type(env_ai_stub)
        
        ai_engine_record = _AI_ENGINE_REGISTRY[ai_engine_name]
        ai_engine = ai_engine_record._create_instance(env_ai_stub, False)
        assert isinstance(ai_engine, AIEngine), type(ai_engine)
        ai_frame = AIFrame(ai_engine)
        
        f = DustFrame()
        f.env = env_frame
        f.ai = ai_frame
        f.env_name = env_name
        f.ai_engine_name = ai_engine_name
        return f

    @staticmethod
    def create_demo_frames(env_name: str, ai_engine_name: str) -> tuple:
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
        env_record = _ENV_REGISTRY[env_name]
        env_core = env_record._create_env()
        assert isinstance(env_core, EnvCore), type(env_core)
        env_frame = EnvFrame(env_core)
        
        env_ai_stub = env_record._create_ai_stub(env_core)
        assert isinstance(env_ai_stub, EnvAIStub), type(env_ai_stub)
        
        ai_engine_record = _AI_ENGINE_REGISTRY[ai_engine_name]
        ai_engine = ai_engine_record._create_instance(env_ai_stub, True)
        assert isinstance(ai_engine, AIEngine), type(ai_engine)
        ai_frame = AIFrame(ai_engine)
        
        env_disp = env_record._create_disp(env_core, env_ai_stub)
        assert isinstance(env_disp, EnvDisplay), type(env_disp)
        disp_frame = DispFrame(env_disp)
        
        f = DustFrame()
        f.env = env_frame
        f.ai = ai_frame
        f.disp = disp_frame
        f.env_name = env_name
        f.ai_engine_name = ai_engine_name
        return f
    
    @staticmethod
    def create_training_frames_from_save(filename):
        with open(filename, 'rb') as f:
            sd = pickle.loads(f.read())
        


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
    
    def state_dict(self) -> dict:
        """ Returns the state dict of the environment.
        """
        logging.info('Exporting environment state')
        return self.env_core.state_dict()
    
    def load_state_dict(self, sd: dict) -> None:
        """ Loads state from a state dict.
        """
        logging.info('Importing environment state')
        raise NotImplementedError()
    
    def next_tick(self):
        self.env_core.next_tick()
        
    def curr_tick(self) -> int:
        return self.env_core.curr_tick()

    def evolve(self) -> None:
        """ Executes one-tick simulation
        """
        # TODO: better name?
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

    def state_dict(self) -> dict:
        """ Returns the state dict of the environment.
        """
        logging.info('Exporting AI engine state')
        return self.ai_engine.state_dict()
    
    def load_state_dict(self, sd: dict) -> None:
        """ Loads state from a state dict.
        """
        logging.info('Importing AI engine state')
        raise NotImplementedError()
    
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
