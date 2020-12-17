import logging

import numpy as np

from dust.core.env import EnvAIStub
from dust.utils import np_utils
from dust.core import progress_log

from dust.ai_engines import prototype

from dust import _dust

BRAIN_NAME = 'brain01'
TERM_NAME = 'term01'
_EPOCH_LENGTH = 200

_ARGPARSER = _dust.argparser()

_ARGPARSER.add_argument('--freeze', action='store_true',
    help='')

class Env01Stub(EnvAIStub):
    
    def __init__(self, project, env, state_dict: dict = None):
        self._proj = project
        if state_dict is not None:
            assert isinstance(state_dict, dict)
            self.curr_epoch_tick = state_dict['curr_epoch_tick']
            self.curr_epoch = state_dict['curr_epoch']
            self.epoch_reward = state_dict['epoch_reward']
            self.epoch_num_rounds = state_dict['epoch_num_rounds']
            engine = prototype.Engine.create_from_state_dict(state_dict['engine'])
        else:
            self.curr_epoch_tick = 0
            self.curr_epoch = 0
            self.epoch_reward = 0 # reward collected in the epoch (NOT round)
            self.epoch_num_rounds = 0
            brain_def = prototype.BrainDef(25, 4, (16, 16))
            engine = prototype.Engine.create_new_instance()
            engine.add_brain(BRAIN_NAME, brain_def)
            engine.add_terminal(TERM_NAME, BRAIN_NAME)
        self.env = env
        self.engine = engine
        self.freeze = self._proj.args.freeze
        if not self.freeze:
            self.progress = progress_log.ProgressLog()
        else:
            self.progress = None
        
    def state_dict(self) -> dict:
        return {'engine': self.engine.state_dict(),
                'curr_epoch_tick': self.curr_epoch_tick,
                'curr_epoch': self.curr_epoch,
                'epoch_reward': self.epoch_reward,
                'epoch_num_rounds': self.epoch_num_rounds}

    @classmethod
    def create_new_instance(cls, project, env_core) -> 'EnvAIStub':
        return cls(project, env_core)
    
    @classmethod
    def create_from_state_dict(cls, project, env_core, state_dict) -> 'EnvAIStub':
        return cls(project, env_core, state_dict)
    
    def perceive_and_act(self) -> None:
        """ Perceives environment state and takes actions
        
        """
        obs = self.engine.create_empty_obs(brain_name=BRAIN_NAME)
        obs['o'] = self._get_observation()
        obs_dict = {TERM_NAME: obs}
        
        exp_dict = self.engine.evaluate(obs_dict)
        exp = exp_dict[TERM_NAME]
        self.env.next_action[:] = exp['act']
        self.exp = exp

    def update(self) -> None:
        self.epoch_reward += self.env.tick_reward
    
        if not self.freeze:
            self.exp['rew'] = self.env.tick_reward
            self.engine.add_experiences({TERM_NAME: self.exp})
            
            end_of_epoch = self.curr_epoch_tick + 1 == _EPOCH_LENGTH
            
            forced_round_end = (not self.env.end_of_round) and end_of_epoch
            
            if self.env.end_of_round or end_of_epoch:
                self.epoch_num_rounds += 1
                self.env.end_of_round = True # Force env to end the round
                if forced_round_end:
                    obs = self.engine.create_empty_obs(brain_name=BRAIN_NAME)
                    obs['o'] = self._get_observation()
                    exp = self.engine.evaluate({TERM_NAME: obs})[TERM_NAME]
                    last_val = exp['val']
                else:
                    last_val = 0
                self.engine.finish_paths({TERM_NAME: last_val})
            
            if end_of_epoch:
                avg_round_reward = self.epoch_reward / self.epoch_num_rounds
                self._proj.log.info('EOE epoch: {} score: {}'.format(
                    self.curr_epoch, avg_round_reward))
                
                fields = self.engine.flush_experiences()[BRAIN_NAME]
                assert self.progress
                self.progress.set_fields(epoch=self.curr_epoch, score=avg_round_reward)
                self.progress.set_fields(**fields)
                self.progress.finish_line()
                
                self.curr_epoch += 1
                self.curr_epoch_tick = 0
                self.epoch_reward = 0
                self.epoch_num_rounds = 0
            else:
                self.curr_epoch_tick += 1
    
    def _get_observation(self):
        """ Extract observation from the current env
        """
        def index_to_coords(idxs, h):
            return np.stack((idxs // h, idxs % h), -1)
        env = self.env
        # Generate vision at each player position
        coords = index_to_coords(env.player_coords, env.map_shape[1])
        map_state = np.zeros(env.map_shape, np.float32)
        map_state_flatten = map_state.reshape(-1)
        map_state_flatten[env.wall_coords] = -1
        map_state_flatten[env.food_coords] = 1
        obs = np_utils.extract_view(map_state, coords, 2, True)
        obs = obs.reshape((len(obs), -1))
        return obs


    
    
