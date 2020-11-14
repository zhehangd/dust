import logging


class _EnvComponent(object):
    
    def __init__(self):
        pass
    
    def state_dict(self):
        """ Returns a state dict storing the state of the object.
        Returns:
            A dict type object
        """
        raise NotImplementedError()
    
    def load_state_dict(self, sd):
        """ Loads a state dict.
        
        Args:
            sd (dict): The state dict.
        """
        raise NotImplementedError()


class EnvCore(_EnvComponent):
    """ Base class of Environment Cores
    
    This is one of the three base classes an environment should implement.
    This one involves the simulation of an environment.
    
    A subclass should implement:
        _create_new_round
        state_dict
        load_state_dict
        
    Attributes:
        curr_tick (int): Ticks since the beginning of the simulation.
        curr_round_tick (int): Ticks since the start of this round.
        curr_round (int): Rounds since the beginning of the simulation.
        tick_reward (int): Reward earned this tick.
        round_reward (int): Reward earned this round so far.
        end_of_round (bool): Indicates the current round is about to end.
    
    """
    
    def __init__(self):
        
        # --------------- Attributes for Read-only -----------
        # They represent the state of the environment for users to check. 
        # Do NOT modify them.
        
        self.curr_tick = 0
        self.curr_round_tick = 0
        self.curr_round = 0
        self.tick_reward = 0
        self.round_reward = 0
        
        # ----------------- Interactive attributes ----------------
        # These attributes are open to read and write in certain conditions
        # to interact with the environment.

        # Flag indicating a round is ended.
        # Refreshed by 'evolve' every tick.
        # 'next_tick' checks this flag and resets the environment if true.
        # One may manually set this flag between 'evolve' and 'next_tick'
        # to trigger the resetting.
        self.end_of_round = False


    def new_simulation(self):
        """ Creates a new simulation
        """
        self._create_new_round()
    
    def evolve(self):
        """ Executes one tick simulation
        """
        self._evolve_tick()
        self.round_reward += self.tick_reward
    
    def next_tick(self):
        self.tick_reward = 0
        self.curr_tick += 1
        self.curr_round_tick += 1
        if self.end_of_round:
            self.end_of_round = False
            self.round_reward = 0
            self.curr_round_tick = 0
            self.curr_round += 1
            self._create_new_round()
 
    def _create_new_round(self):
        """
        Inherited class should implement this
        """
        raise NotImplementedError()
    
    def _evolve_tick(self):
        """
        Inherited class should implement this
        """

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
    
    def __init__(self, env):
        pass
    
    def get_observation(self):
        raise NotImplementedError()
    
    def set_action(self, a):
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
    
    def __init__(self):
        pass
    
    def init(self):
        pass
    
    def render(self):
        pass
