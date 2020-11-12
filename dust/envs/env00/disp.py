import logging

from dust.core.env import EnvDisplay

class Disp(EnvDisplay):
    
    def __init__(self, env, env_ai_stub):
        self.env = env
        self.states = []
        
    def render(self):
        self.states.append(self.env.state)
        if self.env.state in 'GB':
            logging.info(''.join(self.actions))
            self.states = []

    def close(self):
        pass
    
