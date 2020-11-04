import logging

class Disp(object):
    
    def __init__(self, env):
        self.env = env
        self.states = []
        
    def render(self):
        self.states.append(self.env.state)
        if self.env.state in 'GB':
            logging.info(''.join(self.actions))
            self.states = []

    def close(self):
        pass
    
