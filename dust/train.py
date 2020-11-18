import datetime
import logging
import time
import os
import sys

import pickle

import numpy as np

from dust import _dust
from dust.utils import utils
from dust.utils.utils import FindTimestampedFile
from dust.utils.state_dict import show_state_dict_content
from dust.utils.dynamic_save import DynamicSave

class MainLoopTimer(object):
    
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

def init():
    _dust.register_all_envs()
    _dust.register_all_env_arguments()
    _dust.register_all_ai_engines()
    _dust.register_all_ai_engine_arguments()
    proj = _dust.load_project('train')

def train():
    proj = _dust.project()
    
    if proj.args.cont:
        save_filename = FindTimestampedFile('saves', 'save.*.pickle').get_latest_file()
        with open(save_filename, 'rb') as f:
            state_dict = pickle.loads(f.read())
        f = _dust.DustFrame.create_frames(is_train=True, state_dict=state_dict)
    else:
        env_name = proj.args.env
        engine_name = proj.args.engine
        f = _dust.DustFrame.create_frames(env_name=env_name,
                                          ai_engine_name=engine_name,
                                          is_train=True)
        f.env.new_simulation()
    
    save = DynamicSave(f.env.curr_tick(), 1000, 10)
    
    t = MainLoopTimer(0)
    while f.env.curr_tick() < proj.args.target_tick:

        with t.section('env-next-tick'):
            f.env.next_tick()
        
        # Agents observe the environment and take action
        with t.section('agent-act'):
            f.ai.perceive_and_act()
        
        # Environment evolves
        with t.section('env-evolve'):
            f.env.evolve()
        
        # Agents get the feedback from the environment
        # and update themselves
        with t.section('agent-update'):
            f.ai.update()
        
        # Environment update its data and move to the
        # state of the next tick
        with t.section('env-update'):
            f.env.update()
        
        t.finish_iteration()
        if t.time_count >= proj.args.timing_ticks:
            logging.info(t.generate_report_and_reset())
        
        if save.next_update_tick == f.env.curr_tick():
            
            #show_state_dict_content(sd)
            #save_filename = 'saves/{}-{}.pickle'.format(proj.time_tag, f.env.curr_tick())
            save_filename = 'saves/save.{}.pickle'.format(
                datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
            logging.info('Saving to {}'.format(save_filename))
            f.save(save_filename)
            save.add_save(save_filename, f.env.curr_tick())

if __name__ == '__main__':
    
    _argparser = _dust.argparser()

    _argparser.add_argument(
        '--timing_ticks',
        type=int, default=16000,
        help='Number of ticks between each timing')

    _argparser.add_argument(
        '--env', 
        default='env01',
        help='Environment to use')

    _argparser.add_argument(
        '--engine', 
        default='prototype',
        help='AI engine to use')
    
    _argparser.add_argument(
        '--target_tick', 
        type=int, default=50000,
        help='Train until reaching the given tick')
    
    _argparser.add_argument(
        '--cont',
        type=bool, default=False,
        help='Continue training')
    
    try:
        sys.stderr.write('Initializing dust\n')
        init()
        logging.info('Starting training...')
        train()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
