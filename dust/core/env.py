import logging

class EnvCore(object):
    """
    A subclass should implement 
    _create_new_round
    
    """
    
    def __init__(self):
        
        # --------------- Attributes for Read-only -----------
        # They represent the state of the environment for users to check. 
        # Do NOT modify them.
        
        # Tick number since the beginning of the simulation
        self.curr_tick = 0
        
        # Tick number starting from the bginning of the round
        self.curr_round_tick = 0
        
        # Round number
        self.curr_round = 0
        
        # 
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
        logging.info('Creating a new simulation')
        self._create_new_round()
    
    def load_environment(self):
        """ Creates a environment from a save.
        """
        logging.info('Loading a saved environment')
        raise NotImplementedError()
    
    def evolve(self):
        """
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

class EnvAIStub(object):
    
    def __init__(self, env):
        pass
    
    def get_observation(self):
        raise NotImplementedError()
    
    def set_action(self, a):
        raise NotImplementedError()

class EnvDisplay(object):
    
    def __init__(self):
        pass
    
    def init(self):
        pass
    
    def render(self):
        pass
