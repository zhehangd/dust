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
from dust.utils.su_core import create_default_actor_crtic, MLPCategoricalActor, MLPCritic
from dust.utils.exp_buffer import ExpBuffer
from dust.utils.trainer import Trainer

from dust.utils import exp_buffer

_argparser = _dust.argparser()

_BUFFER_CAPACITY = 400
_BUFFER_FLUSH_THRESHOLD = 200

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
        self.buf_size = _BUFFER_CAPACITY
        
    def obs_np2tensor(self, obs):
        return {'o': torch.as_tensor(obs['o'], dtype=torch.float32)}
    
    def act_np2tensor(self, act):
        return {'a': torch.as_tensor(act['a'], dtype=torch.float32)}
    
    def act_tensor2np(self, act):
        shape = tuple(act['a'].shape)
        act_ = np.empty(shape, dtype=[('a', 'i4')])
        act_['a'] = act['a']
        return act_

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
        buf_capacity = brain_def.buf_size
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
    
    def __init__(self, brain_def=None, sd=None):
        
        if sd is not None:
            assert brain_def is None
            brain_def = sd['brain_def']
        
        self.num_obs = brain_def.num_obs
        self.num_acts = brain_def.num_acts
        self.net_size = brain_def.net_size
        
        self.obs_dtype = _make_obs_type(self.num_obs)
        self.act_dtype = _make_act_type()
        self.ext_dtype = _make_ext_type()
        self._brain_def = brain_def
        self.exp_type = exp_buffer.make_exp_frame_dtype(
            self.obs_dtype, self.act_dtype, self.ext_dtype)
        
        pi_model, v_model = create_default_actor_crtic(
            self.num_obs, self.num_acts, self.net_size)
        self.pi_model = pi_model
        self.v_model = v_model
        if sd is not None:
            self.pi_model.load_state_dict(sd['pi_model'])
            self.v_model.load_state_dict(sd['v_model'])
            self.trainer = Trainer.create_from_state_dict(
                self.pi_model, self.v_model, sd['trainer'])
        else:
            self.trainer = Trainer.create_new_instance(
                self.pi_model, self.v_model)
        
    @property
    def brain_def(self):
        return BrainDef(self.num_obs, self.num_acts, self.net_size)
    
    def state_dict(self):
        sd = {}
        sd['brain_def'] = self.brain_def
        sd['pi_model'] = self.pi_model.state_dict()
        sd['v_model'] = self.v_model.state_dict()
        sd['trainer'] = self.trainer.state_dict()
        return sd
    
    @classmethod
    def create_new_instance(cls, brain_def) -> 'Brain':
        brain = cls(brain_def=brain_def)
        return brain
    
    @classmethod
    def create_from_state_dict(cls, sd) -> 'Brain':
        brain_def = sd['brain_def']
        brain = cls(sd=sd)
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
        obs_ts = self._brain_def.obs_np2tensor(obs)
        with torch.no_grad():
            act_dist, _ = self.pi_model(obs_ts)
            if act is not None:
                act_ts = self._brain_def.act_np2tensor(act)
            else:
                act_ts = act_dist.sample()
            assert isinstance(act_ts['a'], torch.Tensor)
            assert act_ts['a'].ndim == 0 or act_ts['a'].ndim == 1
            logp_a = act_dist.log_prob(act_ts)
            val = self.v_model(obs_ts)
        assert act_ts['a'].shape == logp_a.shape
        exp = np.zeros((), dtype=self.exp_type)
        exp['obs'] = obs
        exp['act'] = self._brain_def.act_tensor2np(act_ts)
        exp['ext']['logp'] = logp_a
        exp['val'] = val
        return exp

    def create_empty_obs(self):
        return np.empty((), self.obs_dtype)
    
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
    
    def create_empty_obs(self, term_name=None, brain_name=None):
        assert (term_name is None) + (brain_name is None) == 1
        if term_name:
            brain_name = self.terminals[term_name].brain_name
        assert brain_name is not None
        return self.brains[brain_name].create_empty_obs()
    
    def evaluate(self, obs_dict: dict):
        """ Evaluates observations
        "obs_dict" should be a dict in form of {term_name: obs}.
        Returns {term_name: obs}
        """
        exp_dict = dict()
        for term_name, obs in obs_dict.items():
            assert term_name in self.terminals, \
                '{} not in {}'.format(term_name, self.terminals.keys())
            term = self.terminals[term_name]
            brain_name = term.brain_name
            assert brain_name in self.brains
            brain = self.brains[brain_name]
            exp = brain.evaluate(obs)
            exp_dict[term_name] = exp
        return exp_dict
    
    def add_experiences(self, exp_dict: dict) -> None:
        for term_name, exp in exp_dict.items():
            assert term_name in self.terminals, \
                '{} not in {}'.format(term_name, self.terminals.keys())
            term = self.terminals[term_name]
            term.add_experience(exp)
    
    def finish_paths(self, last_val_dict):
        for term_name, last_val in last_val_dict:
            self.terminals[self.term_name].finish_path(last_val)
        
    
    def flush_experiences(self):
        exp_paths = {} # brain_name : [path1, path2, ...]
        num_terms = 0
        for term_name, term in self.terminals:
            if term.buf.buf_size <= _BUFFER_FLUSH_THRESHOLD:
                continue
            brain_name = term.brain_name
            exp_path = term.get(_BUFFER_FLUSH_THRESHOLD)
            if brain_name not in exp_paths:
                exp_paths[brain_name] = list()
            exp_paths[brain_name].append(exp_path)
            num_terms += 1
        logging.info('Flushed {} terminals'.format(num_terms)) 
        
        for brain_name, paths in exp_paths.items():
            path_length = 0
            for path in paths:
                path_length += len(path)
            # TODO: merge paths and send them to the trainer    
            
            #data = dict(obs=buf_data['obs']['o'], act=buf_data['act']['a'],
            #            logp=buf_data['ext']['logp'], ret=buf_data['ret'],
            #            adv=buf_data['adv'])
            
    
