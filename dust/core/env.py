import logging


class _EnvComponent(object):
    
    def state_dict(self) -> dict:
        """ Returns a state dict storing the state of the object.
        Returns:
            A dict type object
        """
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
    
    def curr_tick(self) -> int:
        raise NotImplementedError()

class EnvAIStub(_EnvComponent):
    """ Base class of AI Stub
    
    This is one of the three base classes an environment should implement.
    This one involves the interaction between an environment and an AI engine.
    
    A subclass should implement:
        get_observation
        set_action
        state_dict
        load_state_dict
        
    Attributes:
    
    """
    
    def get_observation(self) -> None:
        raise NotImplementedError()
    
    def set_action(self, a) -> None:
        raise NotImplementedError()
    
    @property
    def tick_reward(self) -> int:
        raise NotImplementedError()
    
    @property
    def end_of_round(self) -> bool:
        raise NotImplementedError()
    
    @end_of_round.setter
    def end_of_round(self, val) -> None:
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
    
    def init(self):
        raise NotImplementedError()
    
    def render(self):
        raise NotImplementedError()
