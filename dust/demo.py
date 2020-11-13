import logging
import os
import sys
import time

import torch

from dust import _dust

_argparser = _dust.argparser()

_argparser.add_argument(
    '--env', 
    default='env01',
    help='Environment to use')

def init():
    _dust.register_all_envs()
    _dust.register_all_env_arguments()
    proj = _dust.load_project('demo')

def demo():
    
    proj = _dust.project()
    
    env_name = proj.args.env
    env_frame, ai_frame, disp_frame = _dust.create_demo_frames(env_name)
    env_frame.new_simulation()
    
    ai_engine = ai_frame.ai_engine
    
    assert ai_engine.pi_model is not None
    assert ai_engine.v_model is not None
    net_data = torch.load(os.path.join(proj.proj_dir, 'network.pth'))
    ai_engine.pi_model = net_data['pi_model']
    ai_engine.v_model = net_data['v_model']
    
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
        sys.stderr.write('Initializing dust\n')
        init()
        logging.info('Starting simulation...')
        demo()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
