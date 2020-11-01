import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dust.core import project
from dust.utils import utils

if __name__ == '__main__':
    proj = project.load_project()
    
    logging.info('Plotting training progress...')
    
    progress_dir = os.path.join(proj.proj_dir, 'progress')
    find_file = utils.FindTimestampedFile(progress_dir, "*.progress.txt")
    progress_file = find_file.get_latest_file()
    progress_df = pd.read_table(progress_file)

    def smooth_data(x, smooth):
        x = np.asarray(progress_df['score'])
        w = np.ones(smooth, np.float)
        y = np.convolve(x,w,'same') / smooth
        return y
    
    t = np.asarray(progress_df['epoch'])
    y = np.asarray(progress_df['score'])
    z = smooth_data(y, 200)
    print(t.shape, y.shape, z.shape)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(t, z)
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.savefig("plot.png", dpi=72)
