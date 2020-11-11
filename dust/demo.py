import importlib
import logging
import os
import time

import torch

from dust.core import project

from dust import _dust
from dust.utils import utils
from dust.dev.agent import Agent
from dust.core.env import BaseEnv

_argparser = _dust.argparser()

def demo():
    
    proj = _dust.project()
    env_module = importlib.import_module(
        'dust.envs.' + proj.cfg.env + '.core')
    
    env = env_module.Env()
    env.new_environment()
    assert isinstance(env, BaseEnv)
    
    disp_module = importlib.import_module(
        'dust.envs.' + proj.cfg.env + '.disp')
    disp = disp_module.Disp(env)
    
    agent = Agent(env, False)
    assert agent.pi_model is not None
    assert agent.v_model is not None
    net_data = torch.load(os.path.join(proj.proj_dir, 'network.pth'))
    agent.pi_model = net_data['pi_model']
    agent.v_model = net_data['v_model']
    
    disp.render()
    while True:
        agent.act()
        env.evolve()
        agent.update()
        disp.render()
        env.next_tick()
        time.sleep(0.03)

if __name__ == '__main__':
    
    try:
        proj = project.load_project('demo')
        logging.info('Starting simulation...')
        demo()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
