import os
import time

import numpy as np
import torch

from dust.core import project
from dust.dev import simulation

if __name__ == '__main__':
    proj = project.load_project()
    sim = simulation.SimulationDemo(False)
    assert sim.agent.pi_model is not None
    assert sim.agent.v_model is not None
    net_data = torch.load(os.path.join(proj.proj_dir, 'network.pth'))
    sim.agent.pi_model = net_data['pi_model']
    sim.agent.v_model = net_data['v_model']
    sim.start()
