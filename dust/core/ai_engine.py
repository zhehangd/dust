from dust.core.env import EnvCore, EnvAIStub

class AIEngine(object):
    """ Base class of AI engines
    """
    
    def __init__(self):
        """
        """
        pass
    
    @classmethod
    def create_new_instance(cls, ai_stub, **kwargs) -> 'AIEngine':
        """ Creates a new AI Engine instance
        """
        raise NotImplementedError()
    
    @classmethod
    def create_from_state_dict(cls, ai_stub, state_dict, **kwargs) -> 'AIEngine':
        """ Creates an AI Engine instance from a state dict
        """
        raise NotImplementedError()
    
    def perceive_and_act(self) -> None:
        """ Perceives environment state and takes actions
        
        By design called after EnvCore.next_tick and before EnvCore.evolve.
        Typically this function should observe the state of the environment
        and assign the actions taken by the characters in the current tick.
        
        """
        raise NotImplementedError()

    def update(self) -> None:
        raise NotImplementedError()
    
class AIEngineRecord(object):
    """
    """

    @property
    def ai_engine(self) -> type(AIEngine):
        """ Returns the AI Engine class
        """
        raise NotImplementedError()
    
    def register_args(self):
        """ Register arguments of the AI Engine
        """
        raise NotImplementedError()
        
