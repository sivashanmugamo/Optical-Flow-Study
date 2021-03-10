'''
Author: @sivashanmugamo
'''

import cv2
import numpy as np

from video_processing.framePairing import framePairing

cl= framePairing(vid_path= '/Users/shiva/Documents/GitHub/OF/data/test.mp4')
# cl.pairing()
# cl.save(path= '/Users/shiva/Documents/GitHub/OF/data')
# cl.save(path= '')
sh= cl.load(path= 'frame_test.npy')