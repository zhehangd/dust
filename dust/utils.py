import logging
import os
import time

import numpy as np

def extract_view(src, pos, vr, batched=False):
    """
    Args:
      src N-d source array for view extraction.
      pos Coordinates of the view center.
          If batched is False it is an 1-d array of len k, where k <= n.
          If batched is True it is a 2-d array where the 1st dimension is batch
      vr int radius of the view
    """
    if batched:
        bsize = len(pos)
        vsize = len(pos[0])
    else:
        bsize = 1
        vsize = len(pos)
        pos = (pos,)
    
    def make_src_dst_slices(x, w, vr):
        ss = x - vr
        se = x + vr + 1 - w
        ds = de = None
        if ss < 0: ss, ds = ds, -ss
        if se > 0: se, de = de, -se
        if se == 0: se = None
        return slice(ss, se), slice(ds, de)
    
    dst_shape = (bsize,) + (2*vr+1,) * vsize + src.shape[vsize:]
    dst = np.zeros(dst_shape, src.dtype)
    for k in range(bsize):
        src_idxs = []
        dst_idxs = []
        for i in range(vsize):
            p = pos[k][i]
            assert p >= 0 and p < src.shape[i], '{}, {}, {}'.format(p, i, src.shape)
            src_idx, dst_idx = make_src_dst_slices(p, src.shape[i], vr)
            src_idxs.append(src_idx)
            dst_idxs.append(dst_idx)
        src_idxs = tuple(src_idxs)
        dst_idxs = tuple(dst_idxs)
        dst[k][dst_idxs] = src[src_idxs]
    if not batched:
        dst = dst[0]
    return dst

class Timer(object):
    def __init__(self, parent, name):
        self.name = name
        self.parent = parent
        
    def __enter__(self):
        self.t = time.time()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t = time.time() - self.t
        if self.name in self.parent:
            self.parent[self.name] += self.t
        else:
            self.parent[self.name] = self.t
