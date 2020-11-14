import numpy as np

from dust.core.env import EnvAIStub
from dust.utils import np_utils

class Env01Stub(EnvAIStub):
    
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

