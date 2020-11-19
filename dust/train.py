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

class Train(object):
    
    def __init__(self):
        pass
    
    def run(self):
        try:
            self._init()
            logging.info('Starting training...')
            proj = _dust.project()
            while self.sim.env.curr_tick < proj.args.target_tick:
                self._update()
        except KeyboardInterrupt:
            logging.info('Interrupted by user')
    
    def _init(self):
        
        _dust.register_all_envs()
        _dust.register_all_env_arguments()
        _dust.register_all_ai_engines()
        _dust.register_all_ai_engine_arguments()
        proj = _dust.load_project('train')
        
        if proj.args.cont:
            save_filename = FindTimestampedFile('saves', 'save.*.pickle').get_latest_file()
            with open(save_filename, 'rb') as f:
                state_dict = pickle.loads(f.read())
            sim = _dust.DustFrame.create_frames(is_train=True, state_dict=state_dict)
        else:
            env_name = proj.args.env
            engine_name = proj.args.engine
            sim = _dust.DustFrame.create_frames(env_name=env_name,
                                            ai_engine_name=engine_name,
                                            is_train=True)
            sim.env.new_simulation()
        
        self.sim = sim
        self.save = DynamicSave(sim.env.curr_tick, 1000, 10)
        self.timer = MainLoopTimer(0)
    
    def _update(self):
        timer = self.timer
        sim = self.sim
        save = self.save
        proj = _dust.project()
        with timer.section('env-next-tick'):
            sim.env.next_tick()
        
        # Agents observe the environment and take action
        with timer.section('agent-act'):
            sim.ai.perceive_and_act()
        
        # Environment evolves
        with timer.section('env-evolve'):
            sim.env.evolve()
        
        # Agents get the feedback from the environment
        # and update themselves
        with timer.section('agent-update'):
            sim.ai.update()
        
        # Environment update its data and move to the
        # state of the next tick
        with timer.section('env-update'):
            sim.env.update()
        
        timer.finish_iteration()
        if timer.time_count >= proj.args.timing_ticks:
            logging.info(timer.generate_report_and_reset())
        
        if save.next_update_tick == sim.env.curr_tick:
            save_filename = self._save_state()
            save.add_save(save_filename, sim.env.curr_tick)

    def _save_state(self) -> str:
        save_filename = 'saves/save.{}.pickle'.format(
            datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
        logging.info('Saving to {}'.format(save_filename))
        self.sim.save(save_filename)
        return save_filename
    
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
    
    main = Train()
    main.run()
    logging.info('End')
