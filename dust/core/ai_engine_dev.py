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

from dust.utils import exp_buffer

_argparser = _dust.argparser()

_BUFFER_CAPACITY = 200

def _make_obs_type(num_obs):
    return [('o', 'f4', num_obs)]

def _make_act_type():
    return [('a', 'i4')]

def _make_ext_type():
    return [('logp','f4')]

class BrainDef(object):
    """ Description of a brain
    """
    
    def __init__(self, num_obs, num_acts, net_size):
        self.num_obs = num_obs
        self.num_acts = num_acts
        self.net_size = net_size

class Terminal(object):
    
    """ Represents an individual that uses a brain.
    
    It stores the experiences of the individual.
    
    Attributes:
        brain_name (str): Name of the brain used by the terminal.
            The terminal class itself does not use this member.
        buf (exp_buffer.ExpBuffer): The experience buffer.
    
    """
    
    def __init__(self, brain_name, brain_def):
        obs_dtype = _make_obs_type(brain_def.num_obs)
        act_dtype = _make_act_type()
        ext_dtype = _make_ext_type()
        buf_capacity = _BUFFER_CAPACITY
        buf = exp_buffer.ExpBuffer(
            buf_capacity, obs_dtype, act_dtype, ext_dtype)
        self.buf = buf
        self.brain_name = brain_name
    
    def state_dict(self):
        """ Returns a dict that stores the state of the terminal
        Returns:
            A dict storing the terminal state
        """
        return {'brain_name': self.brain_name,
                'buf': self.buf}
    
    def add_experience(self, exp):
        """ Adds an experience frame to the buffer
        
        Parameters:
            exp (np.ndarray):
        """
        self.buf.store(exp)
    
    @classmethod
    def create_new_instance(cls, brain_name, brain_def) -> 'Terminal':
        term = cls(brain_name, brain_def)
        return term
    
    @classmethod
    def create_from_state_dict(cls, sd) -> 'Terminal':
        brain_name = sd['brain_name']
        buf = sd['buf']
        term = cls(brain_name, brain_def)
        return term

class Brain(object):
    """ 
    """
    
    def __init__(self, brain_def):
        self.num_obs = brain_def.num_obs
        self.num_acts = brain_def.num_acts
        self.net_size = brain_def.net_size
        
        self.obs_dtype = _make_obs_type(self.num_obs)
        self.act_dtype = _make_act_type()
        self.ext_dtype = _make_ext_type()
        self.exp_type = exp_buffer.make_exp_frame_dtype(
            self.obs_dtype, self.act_dtype, self.ext_dtype)
        
        pi_model, v_model = create_default_actor_crtic(
            self.num_obs, self.num_acts, self.net_size)
        self.pi_model = pi_model
        self.v_model = v_model
        
    @property
    def brain_def(self):
        return BrainDef(self.num_obs, self.num_acts, self.net_size)
    
    def state_dict(self):
        sd = {}
        sd['brain_def'] = self.brain_def
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
    
    def evaluate(self, obs, act=None):
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
            if act is not None:
                a = torch.as_tensor(act['a'], dtype=torch.int64)
            else:
                a = pi.sample()
            assert isinstance(a, torch.Tensor)
            assert a.ndim == 0 or a.ndim == 1
            logp_a = self.pi_model._log_prob_from_distribution(pi, a)
            val = self.v_model(x)
        assert a.shape == logp_a.shape
        exp = np.zeros((), dtype=self.exp_type)
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
            brain_name, self.brains[brain_name].brain_def)
    
    def remove_terminal(self, term_name):
        del self.terminals[term_name]
    
    def evaluate(self, obs_dict: dict):
        """ Evaluates observations
        "obs_dict" should be a dict in form of {term_name: obs}.
        Returns {term_name: obs}
        """
        exp_dict = dict()
        for term_name, obs in obs_dict:
            assert term_name in self.terminals
            term = self.terminals[term_name]
            brain_name = term.brain_name
            assert brain in self.brains
            brain = self.terminals[brain_name]
            exp = brain.evaluate(obs)
            exp_dict[term_name] = exp
        return exp_dict
    
    def add_experiences(self, exp_dict: dict) -> None:
        #for term_name, exp in data_batch.items():
        #    #assert {'obs', 'act', 'ext', 'rwd'}.issubset(exp.keys())
        for term_name, exp in exp_dict:
            assert term_name in self.terminals
            term = self.terminals[term_name]
            term.add_experience(exp)
    
    def flush_experiences():
        pass
    
    
    
