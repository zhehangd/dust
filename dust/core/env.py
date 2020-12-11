import logging


class _EnvComponent(object):
    
    def state_dict(self) -> dict:
        """ Returns a state dict storing the state of the object.
        Returns:
            A dict type object
        """
        raise NotImplementedError()
    
    @classmethod
    def create_new_instance(cls) -> '_EnvComponent':
        raise NotImplementedError()
    
    @classmethod
    def create_from_state_dict(cls, state_dict) -> '_EnvComponent':
        raise NotImplementedError()

class EnvCore(_EnvComponent):
    """ Base class of Environment Cores
    
    This is one of the three base classes an environment should implement.
    This one involves the simulation of an environment.
        
    Attributes:
        curr_tick (int): Ticks since the beginning of the simulation.
        curr_round_tick (int): Ticks since the start of this round.
        curr_round (int): Rounds since the beginning of the simulation.
        tick_reward (int): Reward earned this tick.
        round_reward (int): Reward earned this round so far.
        end_of_round (bool): Indicates the current round is about to end.
    
    """
    
    @classmethod
    def create_new_instance(cls) -> 'EnvCore':
        raise NotImplementedError()
    
    @classmethod
    def create_from_state_dict(cls, state_dict) -> 'EnvCore':
        raise NotImplementedError()
    
    def new_simulation(self) -> None:
        """ Creates a new simulation
        """
        raise NotImplementedError()
    
    def evolve(self) -> None:
        """ Executes one tick simulation
        """
        raise NotImplementedError()
   
    def update(self) -> None:
        """ Post-evolution update
        """
        raise NotImplementedError()
    
    def next_tick(self) -> None:
        raise NotImplementedError()
    
    @property
    def curr_tick(self) -> int:
        raise NotImplementedError()

class EnvAIStub(_EnvComponent):
    """ Base class of AI Stub
    
    This is one of the three base classes an environment should implement.
    This one involves the interaction between an environment and an AI engine.
    """
    
    @classmethod
    def create_new_instance(cls, env_core) -> 'EnvAIStub':
        raise NotImplementedError()
    
    @classmethod
    def create_from_state_dict(cls, env_core, state_dict) -> 'EnvAIStub':
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

class EnvDisplay(_EnvComponent):
    """ Base class of AI Stub
    
    This is one of the three base classes an environment should implement.
    This one involves the display of the simulation.
    
    A subclass should implement:
        state_dict
        load_state_dict
        
    Attributes:
    
    """
    
    @classmethod
    def create_new_instance(cls, env_core, ai_stub) -> 'EnvDisplay':
        raise NotImplementedError()
    
    @classmethod
    def create_from_state_dict(cls, env_core, ai_stub, state_dict) -> 'EnvDisplay':
        raise NotImplementedError()
    
    def init(self):
        raise NotImplementedError()
    
    def render(self):
        raise NotImplementedError()

class EnvRecord(object):
    """
    """

    @property
    def core(self) -> type(EnvCore):
        raise NotImplementedError()
    
    @property
    def ai_stub(self) -> type(EnvAIStub):
        raise NotImplementedError()
    
    @property
    def disp(self) -> type(EnvDisplay):
        raise NotImplementedError()
    
    def register_args(self):
        raise NotImplementedError()
        
