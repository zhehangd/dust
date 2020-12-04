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

class BrainDef(object):
    
    def __init__(self, num_obs, num_acts, net_size):
        self.num_obs = num_obs
        self.num_acts = num_acts
        self.net_size = net_size

class Terminal(object):
    
    def __init__(self, brain):
        obs_dtype = [('o', 'f4', brain.num_obs)]
        act_dtype = [('a', 'i4')]
        ext_dtype = [('logp','f4')]
        buf_capacity = _BUFFER_CAPACITY
        buf = ExpBuffer(buf_capacity, obs_dtype, act_dtype, ext_dtype)
        self.buf = buf
    
    def state_dict(self):
        return {'term': self}
    
    @classmethod
    def create_new_instance(cls, brain) -> Terminal:
        term = cls(brain)
    
    @classmethod
    def create_from_state_dict(cls, sd) -> Terminal:
        return sd['term']

class Brain(object):
    
    def __init__(self, brain_def):
        self.num_obs = brain_def.num_obs
        self.num_acts = brain_def.num_acts
        self.net_size = brain_def.net_size
        self.brain_def = brain_def
        pi_model, v_model = create_default_actor_crtic(
            self.obs_dim, self.act_dim, self.net_size)
        self.pi_model = pi_model
        self.v_model = v_model
    
    def state_dict(self):
        sd = {}
        sd['brain_def'] = self.brain_def
        sd['pi_model'] = self.pi_model.state_dict()
        sd['v_model'] = self.v_model.state_dict()
        return sd
    
    @classmethod
    def create_new_instance(cls, brain_def) -> AIEngine:
        brain = cls(brain_def)
    
    @classmethod
    def create_from_state_dict(cls, sd) -> AIEngine:
        brain_def = sd['brain_def']
        brain = cls(brain_def)
        brain.pi_model.load_state_dict(sd['pi_model'])
        brain.v_model.load_state_dict(sd['v_model'])
        return brain
        

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
    
    def add_brain(self, name, brain_def) -> int:
        assert name not in self.brains
        self.brains[name] = Brain.create_new_instance(brain_def)
    
    def remove_brain(self, name):
        del self.brains[name]
    
    def add_terminal(self, brain_name):
        self.terminals[name] = self.brains[brain_name]
    
    def remove_terminal(self, name):
        del self.terminals[name]
    
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
    
    
    
