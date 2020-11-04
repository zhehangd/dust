import logging
import os

import torch
import torch.nn as nn

import numpy as np
import scipy.signal

from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from dust import _dust
from dust.core import progress_log
from dust.utils import np_utils

from dust.dev.su_ppo import Context

from dust.envs.env01.core import Env as Env01

_argparser = _dust.argparser()

_argparser.add_argument('--cuda', action='store_true',
    help='Use CUDA')

_EPOCH_LENGTH = 200

class Env00Stub(object):
    
    def __init__(self):
        self.obs_dim = 3
        self.act_dim = 2



class Env01Stub(object):
    
    def __init__(self, env):
        assert isinstance(env, Env01)
        self.env = env
        self.obs_dim = 25
        self.act_dim = 4

    def get_observation(self):
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
     
    def set_action(self, a):
        self.env.next_action[:] = a


class Agent(object):
    
    def __init__(self, env, is_training):
        self.env = env
        self.num_players = 1
        self.env_stub = Env01Stub(env) 
        
        #def count_vars(module):
        #    return sum([np.prod(p.shape) for p in module.parameters()])
        #var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
        #logging.info('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
        
        self.training = is_training
        
        obs_dim = self.env_stub.obs_dim
        act_dim = self.env_stub.act_dim
        self.ctx = Context(obs_dim, act_dim, _EPOCH_LENGTH)
        
        if self.training:
            self.progress = progress_log.ProgressLog()
        
        self.curr_epoch_tick = 0
        self.curr_epoch = 0
        self.epoch_reward = 0 # reward collected in the epoch (NOT round)
        self.num_epoch_collisions = 0
    
    def act(self):
        
        # Check any event and calculate the reward from the last stepb 
        
        # observe
        env = self.env
        
        obs = self.env_stub.get_observation()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        #if ac.use_cuda:
        #    obs_tensor = obs_tensor.cuda()
        a, v, logp = self.ctx.ac.step(obs_tensor)
        
        #self.env.set_action(np.random.randint(0, 4, self.num_players))
        self.env_stub.set_action(a)
        self.ac_data = (a, v, logp, obs)
        #logging.info('act: {}'.format(env.curr_round_tick))

    def update(self):
        # collect rewards
        env = self.env
        
        self.epoch_reward += env.tick_reward
       
        # TODO: ask a stats object from the env/env_stub
        #status_msg = 'tick: {} round: {} round_tick: {} ' \
        #             'epoch: {} epoch_tick: {} epoch_reward: {} round_reward: {} num_collisions: {}'.format(
        #                env.curr_tick, env.curr_round,
        #                env.curr_round_tick, self.curr_epoch,
        #                self.curr_epoch_tick, self.epoch_reward, env.round_reward,
        #                env.num_round_collisions)
        
        if self.training:
            self._update_tick()
            end_of_epoch = self.curr_epoch_tick + 1 == _EPOCH_LENGTH
            
            if env.end_of_round or end_of_epoch:
                # logging.info('end_of_round: ' + status_msg)
                env.end_of_round = True # Force env to end the round
                self._update_round()
                self.num_epoch_collisions += 0#env.num_round_collisions
                
            if end_of_epoch:
                logging.info('end_of_epoch')
                self.progress.set_fields(epoch=self.curr_epoch, score=self.epoch_reward)
                self.progress.set_fields(NumCollisions=self.num_epoch_collisions)
                self.num_epoch_collisions = 0
                self._update_epoch()
                self.progress.finish_line()
                
                
                
            
                if self.curr_epoch % 10 == 0:
                    logging.info('save actor critic')
                    self._save_actor_critic()
                
                self.curr_epoch += 1
                self.curr_epoch_tick = 0
                self.epoch_reward = 0
            else:
                self.curr_epoch_tick += 1
    
    def _update_tick(self):
        a, v, logp, obs = self.ac_data
        r = self.env.tick_reward
        buf = self.ctx.buf
        buf.store(obs, a, r, v, logp)
    
    def _update_round(self):
        obs = self.env_stub.get_observation()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        #if self.ac.use_cuda:
        #    obs_tensor = obs_tensor.cuda()
        _, v, _ = self.ctx.ac.step(obs_tensor)
        self.ctx.buf.finish_path(v)
        
    def _update_epoch(self):
        self.ctx.update(self.progress)
    
    def _save_actor_critic(self):
        proj = _dust.project()
        net_file = os.path.join(proj.proj_dir, 'network.pth')
        torch.save(self.ctx.ac, net_file)
        
