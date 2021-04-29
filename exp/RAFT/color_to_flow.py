import numpy as np
from matplotlib import pyplot as plt
import sys

# np.set_printoptions(threshold=sys.maxsize)

import torch

from PIL import Image

from core.utils import flow_viz

clr_wheel= flow_viz.make_colorwheel()

OF_npy= np.load('/Users/shiva/Documents/GitHub/OF/exp/RAFT/output_imgs/pair_1_pred.npy')

u= OF_npy[0]
v= OF_npy[1]

ncols= clr_wheel.shape[0]

rad= np.sqrt(np.square(u) + np.square(v))
a = np.arctan2(-v, -u)/np.pi
fk = (a+1) / 2*(ncols-1)
k0 = np.floor(fk).astype(np.int32)
k1 = k0 + 1
k1[k1 == ncols] = 0
f= fk-k0

print(f)

for i in range(clr_wheel.shape[1]):
    tmp= clr_wheel[:,i]
    col0= tmp[k0] / 255.0
    col1= tmp[k1] / 255.0

    col= (1-f)*col0 + f*col1

    idx= (rad<=1)

    # print(idx)
    break