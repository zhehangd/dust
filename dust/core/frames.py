class EnvFrame(object):
    """ Environment wrapper
    """
    
    def __init__(self, env_core):
        self.env_core = env_core
    
    def new_simulation(self):
        """ Creates a new simulation
        """
        logging.info('Creating a new simulation')
        self.env_core.new_simulation()

    def load_state_dict(self, sd):
        logging.info('Loading a saved environment')
        
    def evolve(self):
        """ Executes one tick simulation
        """
        self.env_core.evolve()
    
    def update(self):
        self.env_core.next_tick()

class AIFrame(object):
    """ AI Engine wrapper
    """
    
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine
    
    def perceive_and_act(self):
        """ Perceives the environment and takes action
        """
        self.ai_engine.act()

    def update(self):
        """ Receives environment feedback and updates AI
        """
        self.ai_engine.update()

class DispFrame(object):
    """ Display wrapper
    """
    def __init__(self, env_disp):
        self.env_disp = env_disp
    
    def init(self):
        self.env_disp.init()
    
    def render(self):
        self.env_disp.render()
