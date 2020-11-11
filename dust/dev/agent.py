import logging
import os

import torch
import torch.nn as nn

import numpy as np
import scipy.signal

from dust import _dust
from dust.core import progress_log
from dust.utils import np_utils

from dust.core.su_core import create_default_actor_crtic
from dust.core.ppo_buffer import PPOBuffer
from dust.core.trainer import Trainer

# TODO: importing all envs is a bad idea
#from dust.envs.env00.core import Env as Env00
#from dust.envs.env01.core import Env as Env01

_argparser = _dust.argparser()

_argparser.add_configuration(
    '--cuda',
    type=bool, default=False,
    help='Use CUDA')

_EPOCH_LENGTH = 200

class Env00Stub(object):
    
    def __init__(self, env):
        #assert isinstance(env, Env00)
        self.env = env
        self.obs_dim = 3
        self.act_dim = 2
        self.net_size = (3,3)
    
    def get_observation(self):
        state_v = np.zeros(self.obs_dim, np.float32)
        i = ord(self.env.state) - ord('X')
        assert i >= 0 and i < 4, 'state {}'.format(self.env.state)
        state_v[i]
        return state_v
    
    def set_action(self, a):
        self.env.action = 'LR'[a]

class Env01Stub(object):
    
    def __init__(self, env):
        #assert isinstance(env, Env01)
        self.env = env
        self.obs_dim = 25
        self.act_dim = 4
        self.net_size = (16,16)

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

def step(pi_model, v_model, obs, a=None):
    """ Makes one prediction of policy and value
    
    If an action is not given, it takes a random action according to the
    observation and the policy, and returns the selected action, the
    estimated value of the observation, and the logit of which this action
    is selected. If an action is given, then this function behaves as if
    that action is randomly been selected.
    
    This function operates in a no_grad context. 
    """
    with torch.no_grad():
        assert isinstance(obs, torch.Tensor)
        pi = pi_model._distribution(obs)
        a = pi.sample() if a is None else a
        assert isinstance(a, torch.Tensor)
        assert a.ndim == 0 or a.ndim == 1
        logp_a = pi_model._log_prob_from_distribution(pi, a)
        v = v_model(obs)
    assert a.shape == logp_a.shape
    return a, v, logp_a

class Agent(object):
    
    def __init__(self, env, is_training):
        self.env = env
        self.num_players = 1
        
        proj = _dust.project()
        
        if proj.cfg.env == 'env00':
            self.env_stub = Env00Stub(env)
        elif proj.cfg.env == 'env01':
            self.env_stub = Env01Stub(env)
        else:
            raise RuntimeError('Unknown environment '.format(proj.cfg.env))
        #def count_vars(module):
        #    return sum([np.prod(p.shape) for p in module.parameters()])
        #var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
        #logging.info('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
        
        self.training = is_training
        
        obs_dim = self.env_stub.obs_dim
        act_dim = self.env_stub.act_dim
        net_size = self.env_stub.net_size
        
        self.buf = PPOBuffer(obs_dim, None, _EPOCH_LENGTH)
        pi_model, v_model = create_default_actor_crtic(obs_dim, act_dim, net_size)
        self.pi_model = pi_model
        self.v_model = v_model
        self.trainer = Trainer(self.pi_model, self.v_model)
        
        if self.training:
            self.progress = progress_log.ProgressLog()
        
        self.curr_epoch_tick = 0
        self.curr_epoch = 0
        self.epoch_reward = 0 # reward collected in the epoch (NOT round)
        self.epoch_num_rounds = 0
        self.num_epoch_collisions = 0
    
    def act(self):
        
        # Check any event and calculate the reward from the last stepb 
        
        # observe
        env = self.env
        
        obs = self.env_stub.get_observation()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        #if ac.use_cuda:
        #    obs_tensor = obs_tensor.cuda()
        a, v, logp = step(self.pi_model, self.v_model, obs_tensor)
        
        #self.env.set_action(np.random.randint(0, 4, self.num_players))
        self.env_stub.set_action(a)
        self.ac_data = (a, v, logp, obs)
        #logging.info('act: {}'.format(env.curr_round_tick))

    def update(self):
        # collect rewards
        env = self.env
        self.epoch_reward += env.tick_reward
        
        def _update_tick():
            a, v, logp, obs = self.ac_data
            r = self.env.tick_reward
            self.buf.store(obs, a, r, v, logp)
        
        def _update_round():
            if forced_round_end:
                obs = self.env_stub.get_observation()
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                #if self.ac.use_cuda:
                #    obs_tensor = obs_tensor.cuda()
                _, v, _ = step(self.pi_model, self.v_model, obs_tensor)
            else:
                v = 0
            self.buf.finish_path(v)
            
        def _update_epoch():
            pi_info_old, v_info_old, pi_info, v_info = self.trainer.update(self.buf.get())
            kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
            delta_loss_pi = pi_info['loss'] - pi_info_old['loss']
            delta_loss_v = v_info['loss'] - v_info_old['loss']
            self.progress.set_fields(
                LossPi=pi_info_old['loss'], LossV=v_info_old['loss'],
                KL=kl, Entropy=ent, ClipFrac=cf,
                DeltaLossPi=delta_loss_pi,
                DeltaLossV=delta_loss_v)
        
        def _save_actor_critic():
            proj = _dust.project()
            net_file = os.path.join(proj.proj_dir, 'network.pth')
            torch.save({'pi_model': self.pi_model, 'v_model': self.v_model}, net_file)
        
        if self.training:
            _update_tick()
            
            end_of_epoch = self.curr_epoch_tick + 1 == _EPOCH_LENGTH
            
            forced_round_end = (not env.end_of_round) and end_of_epoch
            
            if env.end_of_round or end_of_epoch:
                self.epoch_num_rounds += 1
                # TODO: ask a stats object from the env/env_stub
                avg_round_reward = self.epoch_reward / self.epoch_num_rounds
                status_msg = 'tick: {} round: {} round_tick: {} ' \
                            'epoch: {} epoch_tick: {} round_reward: {} avg_round_reward: {}'.format(
                                env.curr_tick, env.curr_round,
                                env.curr_round_tick, self.curr_epoch,
                                self.curr_epoch_tick, env.round_reward, avg_round_reward)
                logging.info('end_of_round: ' + status_msg)
                env.end_of_round = True # Force env to end the round
                _update_round()
                self.num_epoch_collisions += 0#env.num_round_collisions
            
            if end_of_epoch:
                logging.info('end_of_epoch')
                avg_round_reward = self.epoch_reward / self.epoch_num_rounds
                self.progress.set_fields(epoch=self.curr_epoch, score=avg_round_reward)
                
                #self.progress.set_fields(NumCollisions=self.num_epoch_collisions)
                self.num_epoch_collisions = 0
                _update_epoch()
                self.progress.finish_line()
                
                if self.curr_epoch % 10 == 0:
                    _save_actor_critic()
                    logging.info('saved actor critic')
                
                self.curr_epoch += 1
                self.curr_epoch_tick = 0
                self.epoch_reward = 0
                self.epoch_num_rounds = 0
            else:
                self.curr_epoch_tick += 1
    

        
