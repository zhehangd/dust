import importlib
import logging
import os
import time

import torch

from dust.core import project

from dust import _dust
from dust.utils import utils
from dust.ai_engines.prototype import PrototypeAIEngine
from dust.core.env import EnvCore, EnvAIStub, EnvDisplay

_argparser = _dust.argparser()

def demo():
    
    proj = _dust.project()
    
    env_name = proj.cfg.env
    env_module = importlib.import_module('dust.envs.' + env_name)
    
    env_core = env_module.create_env()
    assert isinstance(env_core, EnvCore), type(env_core)
    env_frame = EnvFrame(env_core)
    
    env_ai_stub = env_module.create_ai_stub(env_core)
    assert isinstance(env_ai_stub, EnvAIStub), type(env_ai_stub)
    
    env_core.new_environment()
    
    env_disp = env_module.create_disp(env_core, env_ai_stub)
    assert isinstance(env_disp, EnvDisplay), type(env_disp)
    disp_frame = DispFrame(env_disp)
    
    agent = PrototypeAIEngine(env_core, env_ai_stub, False)
    ai_frame = AIFrame(agent)
    assert agent.pi_model is not None
    assert agent.v_model is not None
    net_data = torch.load(os.path.join(proj.proj_dir, 'network.pth'))
    agent.pi_model = net_data['pi_model']
    agent.v_model = net_data['v_model']
    
    disp_frame.init()
    disp_frame.render()
    while True:
        ai_frame.perceive_and_act()
        env_frame.evolve()
        ai_frame.update()
        disp_frame.render()
        env_frame.update()
        time.sleep(0.03)

if __name__ == '__main__':
    
    try:
        proj = project.load_project('demo')
        logging.info('Starting simulation...')
        demo()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
