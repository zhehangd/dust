import os
import time

import numpy as np
import torch

from dust.core import project
from dust.dev import simulation

if __name__ == '__main__':
    proj = project.load_project()
    sim = simulation.SimulationDemo(False)
    sim.agent.ac = torch.load(os.path.join(proj.proj_dir, 'network.pth'))
    sim.start()
