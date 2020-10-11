import dust

import numpy as np

import time


def index_to_coords(idxs, h):
    return np.stack((idxs // h, idxs % h), -1)

class Agent(object):
    def __init__(self, env):
        self.env = env
        self.num_players = 1
    
    def step(self):
        
        # Check any event and calculate the reward from the last stepb 
        
        # observe
        env = self.env
        
        map_state = np.zeros(env.map_shape, np.float32)
        map_state_flatten = map_state.reshape(-1)
        map_state_flatten[env.wall_coords] = -1
        map_state_flatten[env.food_coords] = 1
        
        coords = index_to_coords(env.player_coords, env.map_shape[1])
        obs = dust.extract_view(map_state, coords, 2, True)
        print(obs)
        
        self.env.set_action(np.random.randint(0, 4, self.num_players))

class SimulationDemo(object):
    
    def __init__(self):
        self.env = dust.envs.simple.Env()
        self.disp = dust.envs.simple.Disp(self.env)
        self.agent = Agent(self.env)
    
    def start(self):
        """ Start a playing session
        """
        curr_time = 0
        target_time = 1000
        
        #env.load(...) or env.init(...)
        self.disp.render()
        while self.env.curr_time < target_time:
            
            # Agent observes the environment and take action
            self.agent.step()
            time.sleep(60)
            # Environment evolves a step
            self.env.step()
            
            self.disp.render()
            
            
