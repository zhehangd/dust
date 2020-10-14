import numpy as np
import time

from dust.core import project
from dust.dev import simulation

if __name__ == '__main__':
    proj = project.load_project()
    sim = simulation.SimulationDemo(True)
    sim.start()
