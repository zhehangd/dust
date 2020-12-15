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

class Simulation(object):
    
    def __init__(self, env_core, ai_module, disp_module):
        self.env_core = env_core
        self.ai_module = ai_module
        self.disp_module = disp_module
        self.save = DynamicSave(self.env_core.curr_tick, 1000, 10)
        self.timer = MainLoopTimer(0)
    
    @classmethod
    def create_new_instance(cls, env_core_cls, ai_module_cls, disp_module_cls):
        env_core = env_core_cls.create_new_instance()
        ai_module = ai_module_cls.create_new_instance(env_core)
        if disp_module_cls:
            disp_module = create_new_instance(env_core, ai_module)
            disp_module.init()
        else:
            disp_module = None
        return cls(env_core, ai_module, disp_module)
    
    @classmethod
    def create_from_state_dict(cls, env_core_cls, ai_module_cls, disp_module_cls, sd):
        env_core = env_core_cls.create_from_state_dict(sd['env_core'])
        ai_module = ai_module_cls.create_from_state_dict(env_core, state_dict=sd['env_ai_module'])
        if disp_module_cls:
            disp_module = disp_module_cls.create_new_instance(env_core, ai_module)
            disp_module.init()
        else:
            disp_module = None
        return cls(env_core, ai_module, disp_module)
    
    def update(self):
        timer = self.timer
        save = self.save
        proj = _dust.project()
        with timer.section('env-next-tick'):
            self.env_core.next_tick()
        
        # Agents observe the environment and take action
        with timer.section('ai-act'):
            self.ai_module.perceive_and_act()
        
        # Environment evolves
        with timer.section('env-evolve'):
            self.env_core.evolve()
        
        # Agents get the feedback from the environment
        # and update themselves
        with timer.section('ai-update'):
            self.ai_module.update()
        
        # Environment update its data and move to the
        # state of the next tick
        with timer.section('env-update'):
            self.env_core.update()
        
        timer.finish_iteration()
        if timer.time_count >= proj.args.timing_ticks:
            logging.info(timer.generate_report_and_reset())
        
        if save.next_update_tick == self.env_core.curr_tick:
            save_filename = self._save_state()
            save.add_save(save_filename, self.env_core.curr_tick)

        if self.disp_module:
            self.disp_module.render()
            time.sleep(0.03)
        
    def state_dict(self) -> dict:
        sd = {}
        sd['version'] = 'dev'
        sd['env_core'] = self.env_core.state_dict()
        sd['env_ai_module'] = self.ai_module.state_dict()
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
        '--timing_ticks', type=int, default=16000,
        help='Number of ticks between each timing')
    
    _argparser.add_argument(
        '--target_tick', type=int, default=50000,
        help='Train until reaching the given tick')
    
    _argparser.add_argument(
        '--restart', action='store_true',
        help='Continue training')
    
    _argparser.add_argument(
        '--disp', action='store_true',
        help='Demo mode')
    
    proj = _dust.load_project('train')
    proj.parse_args(allow_unknown=True)
    proj.log_proj_info()
    
    _dust.register_all_envs()
    
    assert proj.cfg.env == 'env01'
    env_record = _dust.find_env(proj.cfg.env)
    env_core_class = env_record.import_core()
    env_ai_module_class = env_record.import_ai_module()
    env_disp_module_class = env_record.import_disp_module() if proj.args.disp else None
    
    proj.parse_args() # Parse args again after env modules register their stuff
    
    if proj.args.restart:
        sim = Simulation.create_new_instance(env_core_class, env_ai_module_class, env_disp_module_class)
    else:
        save_filename = FindTimestampedFile('saves', 'save.*.pickle').get_latest_file()
        sd = pickle.loads(open(save_filename, 'rb').read())
        sim = Simulation.create_from_state_dict(env_core_class, env_ai_module_class, env_disp_module_class, sd)
    
    logging.info('Starting training...')
    
    try:
        while sim.env_core.curr_tick < proj.args.target_tick:
            sim.update()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
