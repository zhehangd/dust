import dust

import numpy as np

import time

from dust import agent

import logging

class SimulationDemo(object):
    
    def __init__(self, is_training):
        self.env = dust.envs.simple.Env()
        self.disp = dust.envs.simple.Disp(self.env)
        self.agent = agent.Agent(self.env, is_training)
        self.is_training = is_training
    
    def start(self):
        """ Start a playing session
        """        
        #env.load(...) or env.init(...)
        
        if not self.is_training:
            self.disp.render()
        
        time_count = 0
        time_table = {}
         
        while True:
            
            # Agents observe the environment and take action
            with dust.Timer(time_table, 'agent'):
                self.agent.act()
            
            # Environment evolves
            with dust.Timer(time_table, 'env'):
                self.env.evolve()
            
            # Agents get the feedback from the environment
            # and update themselves
            with dust.Timer(time_table, 'agent'):
                self.agent.update()
            
            # Display the current status
            if not self.is_training:
                self.disp.render()
            
            # Environment update its data and move to the
            # state of the next tick
            with dust.Timer(time_table, 'env'):
                self.env.next_tick()
            
            if not self.is_training:
                time.sleep(0.03)
            
            time_count += 1
            if time_count % 1000 == 0:
                logging.info(' '.join('{}: {} '.format(k, v) for k, v in time_table.items()))
                time_table = {}
            
