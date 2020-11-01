import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dust.core import project

if __name__ == '__main__':
    proj = project.load_project()
    
    progress_file = os.path.join(proj.proj_dir, 'progress/2020-10-20-03-02-22.progress.txt')
    progress_df = pd.read_table(progress_file)

    def smooth_data(x, smooth):
        x = np.asarray(progress_df['score'])
        w = np.ones(smooth, np.float)
        y = np.convolve(x,w,'same') / smooth
        return y
    
    t = np.asarray(progress_df['epoch'])
    y = np.asarray(progress_df['score'])
    z = smooth_data(y, 2000)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(t, z)
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.savefig("plot.png", dpi=72)
