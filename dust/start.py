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
from dust.core.save_mgr import SaveManager

_ARGPARSER = _dust.argparser()

_ARGPARSER.add_configuration('--fps', type=float, default=30.0,
    help='Frame per second. Effective only if display mode is on.')

_ARGPARSER.add_argument('--timing_ticks', type=int, default=16000,
    help='Number of ticks between each timing')

_ARGPARSER.add_argument('--target_tick', type=int, default=50000,
    help='Train until reaching the given tick')

_ARGPARSER.add_argument('--restart', action='store_true',
    help='Continue training')

_ARGPARSER.add_argument('--disp', action='store_true',
    help='Demo mode')

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
    
    def __init__(self, proj, saver, env_core, ai_module, disp_module):
        self.proj = proj
        self.env_core = env_core
        self.ai_module = ai_module
        self.disp_module = disp_module
        self.saver = saver
        self.timer = MainLoopTimer(0)
    
    @classmethod
    def create_new_instance(cls, **kwargs):
        proj = kwargs['project']
        saver = kwargs['saver']
        env_core_cls = kwargs['env_core_cls']
        ai_module_cls = kwargs['ai_module_cls']
        disp_module_cls = kwargs['disp_module_cls']
        env_core = env_core_cls.create_new_instance(proj)
        ai_module = ai_module_cls.create_new_instance(proj, env_core)
        if disp_module_cls:
            disp_module = create_new_instance(proj, env_core, ai_module)
            disp_module.init()
        else:
            disp_module = None
        return cls(proj, saver, env_core, ai_module, disp_module)
    
    @classmethod
    def create_from_state_dict(cls, **kwargs):
        proj = kwargs['project']
        saver = kwargs['saver']
        env_core_cls = kwargs['env_core_cls']
        ai_module_cls = kwargs['ai_module_cls']
        disp_module_cls = kwargs['disp_module_cls']
        state_dict = kwargs['state_dict']
        env_core = env_core_cls.create_from_state_dict(
            proj, state_dict['env_core'])
        ai_module = ai_module_cls.create_from_state_dict(
            proj, env_core, state_dict=state_dict['env_ai_module'])
        if disp_module_cls:
            disp_module = disp_module_cls.create_new_instance(proj, env_core, ai_module)
            disp_module.init()
        else:
            disp_module = None
        return cls(proj, saver, env_core, ai_module, disp_module)
    
    def update(self):
        with self.timer.section('env-next-tick'):
            self.env_core.next_tick()
        
        # Agents observe the environment and take action
        with self.timer.section('ai-act'):
            self.ai_module.perceive_and_act()
        
        # Environment evolves
        with self.timer.section('env-evolve'):
            self.env_core.evolve()
        
        # Agents get the feedback from the environment
        # and update themselves
        with self.timer.section('ai-update'):
            self.ai_module.update()
        
        # Environment update its data and move to the
        # state of the next tick
        with self.timer.section('env-update'):
            self.env_core.update()
        
        self.timer.finish_iteration()
        if self.timer.time_count >= self.proj.args.timing_ticks:
            logging.info(self.timer.generate_report_and_reset())
        
        if self.saver.next_save_tick() == self.env_core.curr_tick:
            self.save()

        if self.disp_module:
            self.disp_module.render()
            time.sleep(1. / self.proj.cfg.fps)
    
    def save(self):
        self.saver.save(self.env_core.curr_tick, self.state_dict())
    
    def state_dict(self) -> dict:
        sd = {}
        sd['version'] = 'dev'
        sd['env_core'] = self.env_core.state_dict()
        sd['env_ai_module'] = self.ai_module.state_dict()
        return sd
    
if __name__ == '__main__':
    
    proj = _dust.load_project(sess_name='train')
    proj.parse_args(allow_unknown=True)
    proj.log_proj_info()
    
    _dust.register_all_envs()
    
    assert proj.cfg.env == 'env01'
    
    sim_kwargs = {}
    sim_kwargs['project'] = proj
    
    env_record = _dust.find_env(proj.cfg.env)
    sim_kwargs['env_core_cls'] = env_record.import_core()
    sim_kwargs['ai_module_cls'] = env_record.import_ai_module()
    sim_kwargs['disp_module_cls'] = env_record.import_disp_module() if proj.args.disp else None
    
    proj.parse_args() # Parse args again after env modules register their stuff
    
    saver = SaveManager(project=proj)
    sim_kwargs['saver'] = saver
    
    state_dict = None
    if not proj.args.restart:
        num_saves = saver.scan_saves()
        if num_saves > 0:
            state_dict = saver.load_latest_save()
        else:
            proj.log.info('Found no existing save, start a new simulation')
    else:
        proj.log.info('Restart a new simulation')
    
    if state_dict:
        sim_kwargs['state_dict'] = state_dict
        sim = Simulation.create_from_state_dict(**sim_kwargs)
    else:
        sim = Simulation.create_new_instance(**sim_kwargs)
    
    logging.info('Starting training...')
    
    try:
        while sim.env_core.curr_tick < proj.args.target_tick:
            sim.update()
        sim.save()
    except KeyboardInterrupt:
        logging.info('Interrupted by user')
    
    logging.info('End')
