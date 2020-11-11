import importlib
import logging
import time

import numpy as np

from dust.core import project

from dust import _dust
from dust.utils import utils
from dust.dev.agent import Agent
from dust.core.env import BaseEnv

_argparser = _dust.argparser()

_argparser.add_configuration(
    '--timing_ticks',
    type=int, default=10000,
    help='Number of ticks between each timing')

_argparser.add_configuration(
    '--env', 
    default='env01',
    help='Environment to use')

class SimTimer(object):
    
    def __init__(self, start_epoch):
        self.time_count = 0
        self.time_table = {}
        self.start_epoch = start_epoch
    
    def section(self, name):
        return utils.Timer(self.time_table, name)

    def finish_iteration(self):
        self.time_count += 1
        
    def generate_report_and_reset(self):
        ts = self.start_epoch
        te = self.start_epoch + self.time_count
        #msg = 'Time {} - {}: '.format(ts, te)
        msg = "TIM "
        msg += ' '.join('<{}>: {:<.4}'.format(k, v) \
            for k, v in self.time_table.items())
        time_table = {}
        self.start_epoch = te
        self.time_count = 0
        return msg

def train():
    proj = _dust.project()
    env_module = importlib.import_module(
        'dust.envs.' + proj.cfg.env + '.core')
    
    env = env_module.Env()
    env.new_environment()
    assert isinstance(env, BaseEnv)
    
    agent = Agent(env, True)
    
    t = SimTimer(0)
    
    while True:
        
        # Agents observe the environment and take action
        with t.section('agent-act'):
            agent.act()
        
        # Environment evolves
        with t.section('env-evolve'):
            env.evolve()
        
        # Agents get the feedback from the environment
        # and update themselves
        with t.section('agent-update'):
            agent.update()
        
        # Environment update its data and move to the
        # state of the next tick
        with t.section('env-nexttick'):
            env.next_tick()
        
        t.finish_iteration()
        if t.time_count >= proj.cfg.timing_ticks:
            logging.info(t.generate_report_and_reset())

if __name__ == '__main__':
    
    try:
        proj = project.load_project('train')
        logging.info('Starting training...')
        train()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
