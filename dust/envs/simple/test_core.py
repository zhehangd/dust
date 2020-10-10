import unittest

import numpy as np

from dust.envs.simple import core

class TestCore(unittest.TestCase):
    
    def setUp(self):
        self.game = core.Core()
        
    def test_a(self):
        game = self.game
        self.assertEqual(game.map_data.ndim, 2)

if __name__ == '__main__':
    unittest.main() 
