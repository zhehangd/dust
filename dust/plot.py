import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dust import _dust
from dust.utils import utils

class YSource(object):
    
    def __init__(self, name, data):
        self.data = data
        self.name = name

if __name__ == '__main__':
    
    _argparser = _dust.argparser()

    _argparser.add_argument('--x', type=str, default='epoch',
                            help='Field for x-axis')

    _argparser.add_argument('--y', type=str, nargs='+', default=['score', 'Entropy'],
                            help='One or more fields mapped to y-axis')

    _argparser.add_argument('--smooth', type=int, default=0,
                            help='Smoothing filter size')

    COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    proj = _dust.load_project('plot')
    proj.parse_args()
    proj.log_proj_info()
    
    logging.info('Plotting training progress...')
    
    progress_dir = os.path.join(proj.proj_dir, 'logs')
    find_file = utils.FindTimestampedFile(progress_dir, "progress.*.txt")
    progress_file = find_file.get_latest_file()
    progress_df = pd.read_table(progress_file)

    def smooth_data(x, smooth):
        w = np.ones(smooth, np.float)
        y = np.convolve(x,w,'same') / smooth
        return y
    
    x_name = proj.args.x
    x = np.asarray(progress_df[x_name])
    smooth_len = min(len(x), proj.args.smooth)
    
    proj.args.x
    
    sources = []
    for i, name in enumerate(proj.args.y):
        y = np.asarray(progress_df[name])
        if smooth_len > 0:
            y = smooth_data(y, smooth_len)
        sources.append(YSource(name, y))
    
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(proj.args.x)
    
    for i, src in enumerate(sources):
        color = COLORS[i % len(sources)]
        ax = ax1 if i == 0 else ax1.twinx()
        ax.set_ylabel(src.name, color=color)
        ln = ax.plot(x, src.data, color=color, label=src.name)
        src.ln = ln
    
    title_line1 = os.path.basename(progress_file)
    title_line2 = x_name + '-' + '/'.join(src.name for src in sources)
    title_lines = title_line1 + '\n' + title_line2
    
    lns = sum([src.ln for src in sources], [])
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    ax1.grid()
    
    plt.title(title_lines)
    
    plot_filename = \
        '{}/plots/plot.{}.png'.format(
        proj.proj_dir, proj.timestamp)
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=72)
    
