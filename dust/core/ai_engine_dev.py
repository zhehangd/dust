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

_argparser = _dust.argparser()

_BUFFER_CAPACITY = 200

def _make_obs_type(num_obs):
    return [('o', 'f4', num_obs)]

def _make_act_type():
    return [('a', 'i4')]

def _make_ext_type():
    return [('logp','f4')]

class BrainDef(object):
    
    def __init__(self, num_obs, num_acts, net_size):
        self.num_obs = num_obs
        self.num_acts = num_acts
        self.net_size = net_size

class Terminal(object):
    
    def __init__(self, brain):
        obs_dtype = _make_obs_type(brain.num_obs)
        act_dtype = _make_act_type()
        ext_dtype = _make_ext_type()
        buf_capacity = _BUFFER_CAPACITY
        buf = ExpBuffer(buf_capacity, obs_dtype, act_dtype, ext_dtype)
        self.buf = buf
    
    def state_dict(self):
        return {'term': self}
    
    @classmethod
    def create_new_instance(cls, brain) -> 'Terminal':
        term = cls(brain)
        return term
    
    @classmethod
    def create_from_state_dict(cls, sd) -> 'Terminal':
        return sd['term']

class Brain(object):
    
    def __init__(self, brain_def):
        self.num_obs = brain_def.num_obs
        self.num_acts = brain_def.num_acts
        self.net_size = brain_def.net_size
        
        self.obs_dtype = _make_obs_type(self.num_obs)
        self.act_dtype = _make_act_type()
        self.ext_dtype = _make_ext_type()
        
        self.buf_dtype = ExpBuffer.generate_elem_dtype(
            self.obs_dtype, self.act_dtype, self.ext_dtype)
        
        pi_model, v_model = create_default_actor_crtic(
            self.num_obs, self.num_acts, self.net_size)
        self.pi_model = pi_model
        self.v_model = v_model
    
    def state_dict(self):
        sd = {}
        sd['brain_def'] = BrainDef(self.num_obs, self.num_acts, self.net_size)
        sd['pi_model'] = self.pi_model.state_dict()
        sd['v_model'] = self.v_model.state_dict()
        return sd
    
    @classmethod
    def create_new_instance(cls, brain_def) -> 'Brain':
        brain = cls(brain_def)
        return brain
    
    @classmethod
    def create_from_state_dict(cls, sd) -> 'Brain':
        brain_def = sd['brain_def']
        brain = cls(brain_def)
        brain.pi_model.load_state_dict(sd['pi_model'])
        brain.v_model.load_state_dict(sd['v_model'])
        return brain
        
    def evaluate(self, obs):
        """ Makes one prediction of policy and value
        
        If an action is not given, it takes a random action according to the
        observation and the policy, and returns the selected action, the
        estimated value of the observation, and the logit of which this action
        is selected. If an action is given, then this function behaves as if
        that action is randomly been selected.
        
        This function operates in a no_grad context. 
        """
        
        assert obs.dtype == self.obs_dtype
        x = torch.as_tensor(obs['o'], dtype=torch.float32)
        with torch.no_grad():
            pi = self.pi_model._distribution(x)
            #a = pi.sample() if a is None else a
            a = pi.sample()
            assert isinstance(a, torch.Tensor)
            assert a.ndim == 0 or a.ndim == 1
            logp_a = self.pi_model._log_prob_from_distribution(pi, a)
            val = self.v_model(x)
        assert a.shape == logp_a.shape
        exp = np.zeros((), dtype=self.buf_dtype)
        exp['obs'] = obs
        exp['act']['a'] = a
        exp['ext']['logp'] = logp_a
        exp['val'] = val
        return exp

class AIEngineDev(object):
    
    def __init__(self):
        self.brains = {}
        self.terminals = {}
    
    @classmethod
    def create_new_instance(cls) -> AIEngine:
        dv = cls()
    
    @classmethod
    def create_from_state_dict(cls, sd) -> AIEngine:
        dv = cls()
        return dv
    
    def add_brain(self, brain_name, brain_def) -> int:
        assert brain_name not in self.brains
        self.brains[brain_name] = Brain.create_new_instance(brain_def)
    
    def remove_brain(self, brain_name):
        del self.brains[brain_name]
    
    def add_terminal(self, term_name, brain_name):
        self.terminals[term_name] = Terminal.create_new_instance(
            self.brains[brain_name])
    
    def remove_terminal(self, term_name):
        del self.terminals[term_name]
    
    def evaluate(self, obs_batch: dict):
        """ Evaluates observations
        "obs_batch" should be a dict in form of {term_name: obs}.
        Returns {term_name: data}
        """
        return {"obs": None, "act": None, "ext": None}
    
    def add_experiences(self, data_batch):
        for term_name, exp in data_batch.items():
            assert {'obs', 'act', 'ext', 'rwd'}.issubset(exp.keys())
        
    
    def flush_experiences():
        pass
    
    
    
