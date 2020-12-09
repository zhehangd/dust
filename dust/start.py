import datetime
import importlib
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

from dust.envs.env01.core import Env01Core
from dust.envs.env01.ai_stub import Env01Stub

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

class Train(object):
    
    def __init__(self, env, ai_stub, demo_mode):
        self.env = env
        self.ai_stub = ai_stub
        self.save = DynamicSave(self.env.curr_tick, 1000, 10)
        self.timer = MainLoopTimer(0)
        self.demo_mode = demo_mode
        if demo_mode:
            disp_module = importlib.import_module('dust.envs.env01.disp')
            self.disp = disp_module.Disp.create_new_instance(env, ai_stub)
            self.disp.init()
    
    @classmethod
    def create_new_instance(cls, env_name, demo_mode):
        env = Env01Core.create_new_instance()
        ai_stub = Env01Stub.create_new_instance(env, freeze=demo_mode)
        train = cls(env, ai_stub, demo_mode)
        train.env.new_simulation()
        return train
    
    @classmethod
    def create_from_state_dict(cls, env_name, demo_mode, sd):
        env = Env01Core.create_from_state_dict(sd['env_core'])
        ai_stub = Env01Stub.create_from_state_dict(env, freeze=demo_mode, state_dict=sd['env_ai_stub'])
        return cls(env, ai_stub, demo_mode)
    
    def update(self):
        timer = self.timer
        save = self.save
        proj = _dust.project()
        with timer.section('env-next-tick'):
            self.env.next_tick()
        
        # Agents observe the environment and take action
        with timer.section('ai-act'):
            self.ai_stub.perceive_and_act()
        
        # Environment evolves
        with timer.section('env-evolve'):
            self.env.evolve()
        
        # Agents get the feedback from the environment
        # and update themselves
        with timer.section('ai-update'):
            self.ai_stub.update()
        
        # Environment update its data and move to the
        # state of the next tick
        with timer.section('env-update'):
            self.env.update()
        
        timer.finish_iteration()
        if timer.time_count >= proj.args.timing_ticks:
            logging.info(timer.generate_report_and_reset())
        
        if self.demo_mode:
            self.disp.render()
            time.sleep(0.03)
        else:
            if save.next_update_tick == self.env.curr_tick:
                save_filename = self._save_state()
                save.add_save(save_filename, self.env.curr_tick)

    def state_dict(self) -> dict:
        sd = {}
        sd['version'] = 'dev'
        sd['env_core'] = self.env.state_dict()
        sd['env_ai_stub'] = self.ai_stub.state_dict()
        return sd
    
    def _save_state(self) -> str:
        save_filename = 'saves/save.{}.pickle'.format(
            datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
        logging.info('Saving to {}'.format(save_filename))
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        with open(save_filename, 'wb') as f:
            pickle.dump(self.state_dict(), f)
        return save_filename
    
if __name__ == '__main__':
    
    _argparser = _dust.argparser()

    _argparser.add_argument(
        '--timing_ticks',
        type=int, default=16000,
        help='Number of ticks between each timing')
    
    _argparser.add_argument(
        '--target_tick', 
        type=int, default=50000,
        help='Train until reaching the given tick')
    
    _argparser.add_argument(
        '--cont',
        type=bool, default=False,
        help='Continue training')
    
    
    _argparser.add_argument(
        '--demo',
        type=bool, default=False,
        help='Demo mode')
    
    env_name = "env01"
    
    proj = _dust.load_project('train')
        
    if proj.args.cont:
        save_filename = FindTimestampedFile('saves', 'save.*.pickle').get_latest_file()
        with open(save_filename, 'rb') as f:
            sd = pickle.loads(f.read())
        sim = Train.create_from_state_dict(env_name, proj.args.demo, sd)
    else:
        sim = Train.create_new_instance(env_name, proj.args.demo)
    
    logging.info('Starting training...')
    
    try:
        while sim.env.curr_tick < proj.args.target_tick:
            sim.update()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
