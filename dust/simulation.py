import dust

import numpy as np

import time

from dust import agent

class SimulationDemo(object):
    
    def __init__(self):
        self.env = dust.envs.simple.Env()
        self.disp = dust.envs.simple.Disp(self.env)
        self.agent = agent.Agent(self.env)
    
    def start(self):
        """ Start a playing session
        """        
        #env.load(...) or env.init(...)
        
        self.disp.render()
         
        while True:
            
            # Agents observe the environment and take action
            self.agent.act()
            
            # Environment evolves
            self.env.evolve()
            
            # Agents get the feedback from the environment
            # and update themselves
            self.agent.update()
            
            # Display the current status
            self.disp.render()
            
            # Environment update its data and move to the
            # state of the next tick
            self.env.next()
            
            time.sleep(0.03)
            
            
