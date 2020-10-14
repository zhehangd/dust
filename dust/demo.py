import numpy as np
import time

import torch

import dust

if __name__ == '__main__':
    proj = dust.create_or_load_project('testproj')
    sim = dust.SimulationDemo(False)
    sim.agent.ac =  torch.load('testprojx/network.pth')
    sim.start()
