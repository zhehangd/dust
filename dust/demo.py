import logging
import os
import sys
import time

import torch

from dust import _dust

from dust.utils.utils import FindTimestampedFile

def init():
    _dust.register_all_envs()
    _dust.register_all_env_arguments()
    _dust.register_all_ai_engines()
    _dust.register_all_ai_engine_arguments()
    proj = _dust.load_project('demo')

def demo():
    
    proj = _dust.project()
    
    save_filename = FindTimestampedFile('saves', 'save.*.pickle').get_latest_file()
    f = _dust.DustFrame.create_demo_frames_from_save(save_filename)
    
    ai_engine = f.ai.ai_engine
    
    assert ai_engine.pi_model is not None
    assert ai_engine.v_model is not None
    net_data = torch.load(os.path.join(proj.proj_dir, 'network.pth'))
    ai_engine.pi_model = net_data['pi_model']
    ai_engine.v_model = net_data['v_model']
    
    f.disp.init()
    f.disp.render()
    while True:
        f.env.next_tick()
        f.ai.perceive_and_act()
        f.env.evolve()
        f.ai.update()
        f.env.update()
        f.disp.render()
        time.sleep(0.03)

if __name__ == '__main__':
    
    _argparser = _dust.argparser()
    
    try:
        sys.stderr.write('Initializing dust\n')
        init()
        logging.info('Starting simulation...')
        demo()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
