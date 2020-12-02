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
        buf_capacity (int): Capacity of the buffer.
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

    def __init__(self, buf_capacity: int,
                 obs_dtype: Union[list, str, np.dtype],
                 act_dtype: Union[list, str, np.dtype],
                 ext_dtype: Union[list, str, np.dtype],
                 gamma: float = 0.99, lam: float = 0.95):
        buf_dtype = [
            ('obs', obs_dtype), ('act', act_dtype), ('ext', ext_dtype),
            ('rew', 'f4'),  ('val', 'f4'), ('adv', 'f4'), ('ret', 'f4')]
        self.buf_data = np.zeros(buf_capacity, buf_dtype)
        self.gamma = gamma
        self.lam = lam
        self._buf_idx = 0
        self._path_start_idx = 0

    @property
    def buf_capacity(self) -> int:
        return self.buf_data.size
    
    @property
    def buf_size(self) -> int:
        return self._buf_idx
    
    @property
    def path_start_idx(self) -> int:
        return self._path_start_idx

    def store(self, obs, act, ext, rew: float, val: float) -> None:
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
        """ Ends the current path
        
        Call this at the end of trajectory. To properly estimate the expected
        returns, the method needs the value of the state resulted from the
        last action. 
        
        The method looks back in the buffer to where the trajectory started,
        and uses rewards and value estimates from the whole trajectory to
        compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        self._update_adv_and_ret(last_val)
        self._path_start_idx = self._buf_idx

    def get(self, length: int = 0) -> dict:
        """ Retrieve buffer data
        
        "length" specifies the length of data to be retrieved.
        If "length" is not a positive number, it means the whole buffer.
        
        Call this only if there are enough data in the buffer.
        Either there are at least 'length' elements and all covered paths
        have been finished (by calling "finish_path"),  or there are at least
        "length+1" elements, so advantages and returns can be properly estimated.
        
        The retrieved data are removed from the buffer.
        
        """
        num_retrieved = length if length > 0 else self.buf_data.size
        num_remains = self.buf_data.size - num_retrieved
        
        assert num_retrieved <= self._buf_idx, \
            'Buffer size/capacity={}/{} cannot fill a slice of length {}'.format(
                self._buf_idx, self.buf_data.size, num_retrieved)
        
        if num_retrieved > self._path_start_idx and num_retrieved < self._buf_idx - 1:
            self._update_adv_and_ret(self.buf_data['val'][num_retrieved])
        else:
            assert num_retrieved <= self._path_start_idx, \
                'You must finish the current path start/size/capacity={}/{}/{} '\
                'or provide at least one more sample to retrieve your slice of length {}'.format(\
                    self._path_start_idx, self._buf_idx, self.buf_data.size, num_retrieved)
        
        buf_data_ret = self.buf_data[:num_retrieved].copy()
        self.buf_data[:num_remains] = self.buf_data[num_retrieved:]
        self._buf_idx -= num_retrieved
        self._path_start_idx -= num_retrieved
        return buf_data_ret
 
    def _update_adv_and_ret(self, last_val: float = 0) -> None:
        """ Updates the returns and the advantages of the current path
        
        This does all the job to finish a path except that the path is not
        truly finished. It can still receive new data and finally get finished
        by "finish_path".
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
