import numpy as np

from dust.core.env import EnvAIStub

class Env00Stub(EnvAIStub):
    
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
