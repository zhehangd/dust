from typing import Union

import numpy as np
import torch

import numpy as np

from dust.utils import su_core as core

# V1 PPOBuffer was directly copied from spinning up, which has several
# limitations that don't satisfy our needs. So here I am going to replace
# it.
#
# Feature:
# * Retrieving data before the buffer is completely full is allowed.
#   In a dynamic environment, agents can be born at any time so filling
#   their buffers at the same time makes no sense.
# * Allows arbitrary observation/action type. Specifically, supports
#   using structured arrays for obs/act.
# * Save/load

class ExpBuffer(object):
    """
    A buffer for storing trajectories experienced by an agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    
    Args:
        buf_size (int): Size of the buffer.
            Data must be retrieved before the buffer is full.
        obs_dtype (list/str/np.dtype): Dtype of observation.
            Typically a structured type.
        act_dtype (list/str/np.dtype): Dtype of observation. 
            Typically a structured type.
        ext_dtype (list/str/np.dtype): Dtype of extra data. 
            Typically a structured type.
        gamma (float): Discount factor of return
        lam (float): Discount factor of advantage
    
    Attributes:
    
        buf_data (np.ndarray): A 1D structured array holding all buffer data.
            It has 7 components: observation ``obs``, action ``act``, advantage
            ``adv``, reward ``rew``, return ``ret``, value ``val``, action
            extra ``ext``. ``obs``, ``act``, ``ext`` use custom types.
            ``adv``, ``rew``, ``ret``, and ``val`` are float scalars.
        gamma (float): Discount factor of return
        lam (float): Discount factor of advantage
    
        
    Examples:
    
        >>> buf = ExpBuffer(42, [('o1', 'i4', 4), ('o2', 'f4')], [('a', 'i4')])
    
    """

    def __init__(self, buf_size: int,
                 obs_dtype: Union[list, str, np.dtype],
                 act_dtype: Union[list, str, np.dtype],
                 ext_dtype: Union[list, str, np.dtype],
                 gamma: float = 0.99, lam: float = 0.95):
        buf_dtype = [('obs', obs_dtype), ('act', act_dtype), ('ext', ext_dtype),
                     ('adv', 'f4'), ('rew', 'f4'), ('ret', 'f4'),
                     ('val', 'f4'), ]
        self.buf_data = np.zeros(buf_size, buf_dtype)
        self.gamma = gamma
        self.lam = lam
        self._buf_idx = 0
        self._path_start_idx = 0

    def store(self, obs: np.ndarray, act: np.ndarray, ext: np.ndarray,
              rew: float, val: float) -> None:
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self._buf_idx < self.buf_data.size     # buffer has to have room so you can store
        buf_elem = self.buf_data[self._buf_idx]
        buf_elem['obs'] = obs
        buf_elem['act'] = act
        buf_elem['ext'] = ext
        buf_elem['rew'] = rew
        buf_elem['val'] = val
        self._buf_idx += 1

    def finish_path(self, last_val: float = 0) -> None:
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        buf_slice = self.buf_data[slice(self._path_start_idx, self._buf_idx)]
        
        rews = buf_slice['rew']
        rews = np.append(rews, last_val)
        vals = buf_slice['val']
        vals = np.append(vals, last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        # Deltas: difference between observed return and expected return
        #   d(t) = r(t) + \gamma * V(t+1) - V(t)   
        # 
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        buf_slice['adv'] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        buf_slice['ret'] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self._path_start_idx = self._buf_idx

    def get(self, length: int = 0) -> dict:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        num_retrieved = length if length > 0 else self.buf_data.size
        num_remains = self.buf_data.size - num_retrieved
        
        assert self._buf_idx >= num_retrieved, \
            'Buffer does not have enough data {}/{}/{}'.format(
                self._buf_idx, num_retrieved, self.buf_data.size)
        assert self._path_start_idx >= self._buf_idx, \
            'You must finish a path before you retrieve the buffer data'
        
        buf_data_ret = self.buf_data[:num_retrieved].copy()
        self.buf_data[:num_remains] = self.buf_data[num_retrieved:]
        self._buf_idx -= num_retrieved
        self._path_start_idx -= num_retrieved
        return buf_data_ret
 
