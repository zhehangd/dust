import dust

import numpy as np

import time

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.num_players = 1
    
    def step(self):
        # observe
        self.env.set_action(np.random.randint(0, 4, self.num_players))

class SimulationDemo(object):
    
    def __init__(self):
        self.env = dust.envs.simple.Env()
        self.disp = dust.envs.simple.Disp(self.env)
        self.agent = Agent(self.env)
    
    def start(self):
        """ Start a playing session
        """
        curr_time = 0
        target_time = 1000
        
        #env.load(...) or env.init(...)
        while self.env.curr_time < target_time:
            
            # Agent observes the environment and take action
            self.agent.step()
            
            # Environment evolves a step
            self.env.step()
            
            self.disp.render()
            time.sleep(0.1)
            
