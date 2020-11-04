import logging

from dust.core.env import BaseEnv

# There is a good ending state (G) and a bad ending state (B)
# There are three states X, Y, Z and two actions L and R.
# Transition
#    X    Y    Z
# L  Y    B    G
# R  G    Z    B
#

class Env(BaseEnv):
    
    """
    
    Attributes:
    
    
    """
    
    def __init__(self):
        pass

    def _create_new_round(self):
        
        self.state = 'X'
        
        self.action = 'L'
       
        self.lut = {
            'X': {'L': 'Y', 'R': 'G'},
            'Y': {'L': 'B', 'R': 'Z'},
            'Z': {'L': 'G', 'R': 'B'},
        }

    def _evolve_tick(self):
        self.state = self.lut[self.state][self.action] 
        if self.state == 'G':
            self.tick_reward += 10 
            self.end_of_round = True
        if self.state == 'B':
            self.tick_reward -= 10 
            self.end_of_round = True
 
