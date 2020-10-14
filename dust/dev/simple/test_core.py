import unittest

import numpy as np

from . import core

class TestCore(unittest.TestCase):
    
    def setUp(self):
        self.game = core.Env()
        
    

if __name__ == '__main__':
    unittest.main() 
