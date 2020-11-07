import importlib
import logging
import time

import numpy as np

from dust import _dust
from dust.utils import utils
from dust.dev import agent
from dust.core.env import BaseEnv

_argparser = _dust.argparser()

_argparser.add_argument('--timing_ticks', type=int, default=10000,
                        help='Number of ticks between each timing')

_argparser.add_argument('--env', default='env01',
                        help='Environment to use')

class SimulationDemo(object):
    
    def __init__(self, is_training):
        proj = _dust.project()
        env_module = importlib.import_module('dust.envs.' + proj.args.env + '.core')
        
        
        self.env = env_module.Env()
        self.env.new_environment()
        
        if not is_training:
            disp_module = importlib.import_module('dust.envs.' + proj.args.env + '.disp')
            self.disp = disp_module.Disp(self.env)
        self.agent = agent.Agent(self.env, is_training)
        self.is_training = is_training
        
        
    
    def start(self):
        """ Start a playing session
        """        
        #env.load(...) or env.init(...)
        
        proj = _dust.project()
        logging.info('proj.args.timing_ticks={}'.format(proj.args.timing_ticks))
        
        if not self.is_training:
            self.disp.render()
        
        time_count = 0
        time_table = {}
        
        assert isinstance(self.env, BaseEnv)
         
        while True:
            
            # Agents observe the environment and take action
            with utils.Timer(time_table, 'agent1'):
                self.agent.act()
            
            # Environment evolves
            with utils.Timer(time_table, 'env'):
                self.env.evolve()
            
            # Agents get the feedback from the environment
            # and update themselves
            with utils.Timer(time_table, 'agent2'):
                self.agent.update()
            
            # Display the current status
            if not self.is_training:
                self.disp.render()
            
            # Environment update its data and move to the
            # state of the next tick
            with utils.Timer(time_table, 'env'):
                self.env.next_tick()
            
            if not self.is_training:
                time.sleep(0.03)
            
            time_count += 1
            if time_count % proj.args.timing_ticks == 0:
                msg = 'Time cost in {} ticks: '.format(proj.args.timing_ticks)
                msg += ' '.join('<{}>: {:<.4}'.format(k, v) \
                    for k, v in time_table.items())
                logging.info(msg)
                time_table = {}
            
