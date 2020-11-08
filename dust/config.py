import logging
import os
import time

import numpy as np
import torch

from dust.core import project
from dust.dev import simulation

"""
Saves args to project file
""" 

if __name__ == '__main__':
    proj = project.load_project('config')
    proj.save_project()
