import logging

class BaseEnv(object):
    """
    A subclass should implement 
    _create_new_round
    
    """
    
    def __init__(self):
        
        # Tick number since the beginning of the simulation
        self.curr_tick = 0
        
        # Tick number starting from the bginning of the round
        self.curr_round_tick = 0
        
        # Round number
        self.curr_round = 0
        
        self.obs_dim = 3
        
        self.act_dim = 2
        
        # Flag indicating a round is ended.
        # Refreshed by 'evolve' every tick.
        # 'next_tick' checks this flag and resets the environment if true.
        # One may manually set this flag between 'evolve' and 'next_tick'
        # to trigger the resetting.
        self.end_of_round = False
        
        # 
        self.tick_reward = 0
        
        self.round_reward = 0
    
    def new_environment(self):
        """ Creates a new environment.
        """
        logging.info('Creating a new environment')
        self._create_new_round()
    
    def load_environment(self):
        """ Creates a environment from a save.
        """
        logging.info('Loading a saved environment')
        raise NotImplementedError()
    
    def evolve(self):
        """
        Inherited class should implement this
        """
        raise NotImplementedError()
    
    def next_tick(self):
        self.curr_tick += 1
        self.curr_round_tick += 1
        if self.end_of_round:
            self.end_of_round = False
            self.curr_round += 1
            self._create_new_round()
 
    def _create_new_round(self):
        """
        Inherited class should implement this
        """
        raise NotImplementedError()
