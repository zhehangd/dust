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
    
    Attributes:
    
        buf_data (np.ndarray): A structured array holding all buffer data
        gamma (float): Discount factor of returns
        lam (float): Discount factor of advantages
    """

    def __init__(self, buf_size, obs_dtype, act_dtype, gamma=0.99, lam=0.95):
        
        #obs_dtype = [('a', 'i4', 4), ('b', 'f4', 2)]
        #act_dtype = [('x', 'i4')]
        buf_dtype = [('obs', obs_dtype), ('act', act_dtype),
                     ('adv', 'f4'), ('rew', 'f4'), ('ret', 'f4'),
                     ('val', 'f4'), ('logp', 'f4')]
        self.buf_data = np.zeros(buf_size, buf_dtype)
        self.gamma = gamma
        self.lam = lam
        self._buf_idx = 0
        self._path_start_idx = 0

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float,
              val: float, logp: float) -> None:
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self._buf_idx < self.buf_data.size     # buffer has to have room so you can store
        buf_elem = self.buf_data[self._buf_idx]
        buf_elem['obs'] = obs
        buf_elem['act'] = act
        buf_elem['rew'] = rew
        buf_elem['val'] = val
        buf_elem['logp'] = logp
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

        path_slice = slice(self._path_start_idx, self._buf_idx)
        
        buf_slice = self.buf_data[path_slice]
        
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

    def get(self) -> dict:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self._buf_idx == self.buf_data.size, '{}/{}'.format(self._buf_idx, self.buf_data.size) # buffer has to be full before you can get
        self._buf_idx, self._path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # NOTE: NO MPI
        #adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        #self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.buf_data['obs'], act=self.buf_data['act'], ret=self.buf_data['ret'],
                    adv=self.buf_data['adv'], logp=self.buf_data['logp'])
        #return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        return data
 
