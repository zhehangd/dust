import logging
import os

import torch
import torch.nn as nn

import numpy as np
import scipy.signal

from dust import _dust
from dust.core.ai_engine import AIEngine
from dust.core import progress_log
from dust.utils import np_utils
from dust.utils.su_core import create_default_actor_crtic
from dust.utils.exp_buffer import ExpBuffer
from dust.utils.trainer import Trainer
from dust.utils.state_dict import auto_make_state_dict, auto_load_state_dict

_argparser = _dust.argparser()

_argparser.add_configuration(
    '--cuda',
    type=bool, default=False,
    help='Use CUDA')

_EPOCH_LENGTH = 200

def step(pi_model, v_model, obs):
    """ Makes one prediction of policy and value
    
    If an action is not given, it takes a random action according to the
    observation and the policy, and returns the selected action, the
    estimated value of the observation, and the logit of which this action
    is selected. If an action is given, then this function behaves as if
    that action is randomly been selected.
    
    This function operates in a no_grad context. 
    """
    with torch.no_grad():
        #assert isinstance(obs, torch.Tensor)
        act_dist = pi_model(obs)
        a = act_dist.sample()
        assert isinstance(a, dict)
        assert isinstance(a['a'], torch.Tensor)
        assert a['a'].ndim == 0 or a['a'].ndim == 1
        logp_a = act_dist.log_prob(a)
        v = v_model(obs)
    assert a['a'].shape == logp_a.shape
    return a, v, logp_a

class PrototypeAIEngine(AIEngine):
    
    _STATE_DICT_ATTR_LIST = [
            'pi_model', 'v_model', 'buf',
            'curr_epoch_tick', 'curr_epoch', 'epoch_reward', 'epoch_num_rounds']
    
    def __init__(self, ai_stub, freeze, state_dict=None):
        self.ai_stub = ai_stub
        proj = _dust.project()
        
        self.freeze_learning = freeze
        
        obs_dim = self.ai_stub.obs_dim
        act_dim = self.ai_stub.act_dim
        net_size = self.ai_stub.net_size
        
        pi_model, v_model = create_default_actor_crtic(obs_dim, act_dim, net_size)
        self.pi_model = pi_model
        self.v_model = v_model
        
        if not self.freeze_learning:
            self.progress = progress_log.ProgressLog()
        
        if state_dict:
            self.pi_model.load_state_dict(state_dict['pi_model'])
            self.v_model.load_state_dict(state_dict['v_model'])
            self.buf = state_dict['buf']
            self.curr_epoch_tick = state_dict['curr_epoch_tick']
            self.curr_epoch = state_dict['curr_epoch']
            self.epoch_reward = state_dict['epoch_reward']
            self.epoch_num_rounds = state_dict['epoch_num_rounds']
            self.trainer = Trainer.create_from_state_dict(
                self.pi_model, self.v_model, state_dict['trainer'])
        else:
            self.buf = ExpBuffer(_EPOCH_LENGTH, [('o', 'f4', obs_dim)],
                                 [('a', 'i4')], [('logp','f4')])
            self.curr_epoch_tick = 0
            self.curr_epoch = 0
            self.epoch_reward = 0 # reward collected in the epoch (NOT round)
            self.epoch_num_rounds = 0
            self.trainer = Trainer.create_new_instance(
                self.pi_model, self.v_model)
    
    @classmethod
    def create_new_instance(cls, ai_stub, **kwargs) -> AIEngine:
        freeze = kwargs['freeze'] if hasattr(kwargs, 'freeze') else False
        return cls(ai_stub, freeze)
    
    @classmethod
    def create_from_state_dict(cls, ai_stub, state_dict, **kwargs) -> AIEngine:
        freeze = kwargs['freeze'] if hasattr(kwargs, 'freeze') else False
        return cls(ai_stub, freeze, state_dict)
    
    def perceive_and_act(self):
        
        obs = self.ai_stub.get_observation()
        obs_ = {'o': torch.as_tensor(obs, dtype=torch.float32)}
        a, v, logp = step(self.pi_model, self.v_model, obs_)
        
        self.ai_stub.set_action(a['a'])
        self.ac_data = (a['a'], v, logp, obs)
    
    def update(self):
        ai_stub = self.ai_stub
        self.epoch_reward += ai_stub.tick_reward
        
        def _update_tick():
            a, v, logp, obs = self.ac_data
            r = self.ai_stub.tick_reward
            obs_ = np.empty((),[('o','f4',self.ai_stub.obs_dim)])
            a_ = np.empty((),[('a','f4')])
            e_ = np.empty((),[('logp','f4')])
            obs_['o'] = obs
            a_['a'] = a
            e_['logp'] = logp
            self.buf.store(self.buf.create_frame(
                obs_, a_, e_, r, v))
            del self.ac_data
        
        def _update_round():
            if forced_round_end:
                obs = self.ai_stub.get_observation()
                obs_ = {'o': torch.as_tensor(obs, dtype=torch.float32)}
                _, v, _ = step(self.pi_model, self.v_model, obs_)
            else:
                v = 0
            self.buf.finish_path(v)
            
        def _update_epoch():
            buf_data = self.buf.get()
            data = dict(
                obs={'o': torch.as_tensor(buf_data['obs']['o'], dtype=torch.float32)},
                act={'a': torch.as_tensor(buf_data['act']['a'], dtype=torch.float32)},
                #act=torch.as_tensor(buf_data['act']['a'], dtype=torch.float32),
                logp=torch.as_tensor(buf_data['ext']['logp'], dtype=torch.float32),
                ret=torch.as_tensor(buf_data['ret'], dtype=torch.float32),
                adv=torch.as_tensor(buf_data['adv'], dtype=torch.float32))
            pi_info_old, v_info_old, pi_info, v_info = self.trainer.update(data)
            kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
            delta_loss_pi = pi_info['loss'] - pi_info_old['loss']
            delta_loss_v = v_info['loss'] - v_info_old['loss']
            self.progress.set_fields(
                LossPi=pi_info_old['loss'], LossV=v_info_old['loss'],
                KL=kl, Entropy=ent, ClipFrac=cf,
                DeltaLossPi=delta_loss_pi,
                DeltaLossV=delta_loss_v)
        
        if not self.freeze_learning:
            _update_tick()
            
            end_of_epoch = self.curr_epoch_tick + 1 == _EPOCH_LENGTH
            
            forced_round_end = (not ai_stub.end_of_round) and end_of_epoch
            
            if ai_stub.end_of_round or end_of_epoch:
                self.epoch_num_rounds += 1
                ai_stub.end_of_round = True # Force env to end the round
                _update_round()
            
            if end_of_epoch:
                avg_round_reward = self.epoch_reward / self.epoch_num_rounds
                logging.info('EOE epoch: {} score: {}'.format(self.curr_epoch, avg_round_reward))
                self.progress.set_fields(epoch=self.curr_epoch, score=avg_round_reward)
                
                _update_epoch()
                self.progress.finish_line()
                
                self.curr_epoch += 1
                self.curr_epoch_tick = 0
                self.epoch_reward = 0
                self.epoch_num_rounds = 0
            else:
                self.curr_epoch_tick += 1
    
    def state_dict(self) -> dict:
        sd = auto_make_state_dict(self, self._STATE_DICT_ATTR_LIST)
        sd['trainer'] = self.trainer.state_dict()
        return sd
