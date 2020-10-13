import numpy as np
import time

import dust

if __name__ == '__main__':
    proj = dust.create_or_load_project('testproj')
    sim = dust.SimulationDemo()
    sim.train()
