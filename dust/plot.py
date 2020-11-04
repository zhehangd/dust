import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dust import _dust
from dust.core import project
from dust.utils import utils

_argparser = _dust.argparser()

_argparser.add_argument('--x', type=str, default='epoch',
                        help='Field for x-axis')

_argparser.add_argument('--y', type=str, default='score',
                        help='Field for y-axis')

_argparser.add_argument('--smooth', type=int, default=0,
                        help='Smoothing filter size')


if __name__ == '__main__':
    proj = project.load_project()
    
    logging.info('Plotting training progress...')
    
    progress_dir = os.path.join(proj.proj_dir, 'logs')
    find_file = utils.FindTimestampedFile(progress_dir, "progress.*.txt")
    progress_file = find_file.get_latest_file()
    progress_df = pd.read_table(progress_file)

    def smooth_data(x, smooth):
        w = np.ones(smooth, np.float)
        y = np.convolve(x,w,'same') / smooth
        return y
    
    t = np.asarray(progress_df[proj.args.x])
    y = np.asarray(progress_df[proj.args.y])
    
    smooth_len = min(len(t), proj.args.smooth)
    if smooth_len > 0:
        y = smooth_data(y, smooth_len)
    
    fig = plt.figure(figsize=(12, 8))
    plt.plot(t, y)
    plt.grid(True)
    plt.xlabel(proj.args.x)
    plt.ylabel(proj.args.y)
    plt.savefig("plot.png", dpi=72)
