'''
Author: @sivashanmugamo

Run the following command:
python main.py --model_dir [model path] --input_dir [input images path] --output_dir [output imaegs path]
python main.py --model_dir exp/RAFT/models/raft-kitti.pth --input_dir exp/RAFT/input_imgs --output_dir exp/RAFT/output_imgs
'''

# Importing required libraries
import argparse
import os, sys, math, random, pickle
from collections import deque, OrderedDict, defaultdict
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import cv2

from PIL import Image

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F

from core.raft import RAFT
from core.utils.utils import InputPadder

# Global initiations
DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(img_path: str) -> None:
    '''
    '''
    # Reads HWC (Height x Width x Channel)
    img_data= cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_numpy= cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

    print(img_numpy.dtype)

    img_numpy= np.array(Image.open(img_path)).astype(np.int8)
    
    # Changes order to support PyTorch (CHW)
    img_tensor= torch.from_numpy(img_numpy).permute(2, 0, 1).float()
    
    return img_tensor

def main(args) -> None:
    '''
    '''

    input_dir= args.input_dir
    model_dir= args.model_dir
    output_dir= args.output_dir

    # Implementing parallelism at a module level
    model= torch.nn.DataParallel(RAFT(args))

    loaded_model= torch.load(model_dir, map_location= DEVICE)
    model.load_state_dict(loaded_model)

    model.to(device= DEVICE)
    model.eval()

    for r, d, f in os.walk(input_dir):
        img_list= [load_image(os.path.join(r, each_img)) for each_img in sorted(f)]

    img_list= [img_list[n:n+2] for n in range(0, len(img_list), 2)]
    
    with torch.no_grad():
        for each_pair in img_list:
            frame_1, frame_2= each_pair
            # img_padder= InputPadder(frame_1.shape, mode= 'kitti')
            # frame_1, frame_2= img_padder.pad(frame_1, frame_2)

            h, w= frame_1.shape[-2:]
            pad_h= (((h//8)+1)*8-h)%8
            pad_w= (((w//8)+1)*8-w)%8
            pad= [pad_w//2, pad_w - pad_w//2, 0, pad_h]
            print(2*len(frame_1.shape))
            print(pad)

            frame_1= F.pad(each_pair[0], pad, mode= 'replicate')

            print(frame_1.shape, frame_2.shape)

if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--model_dir', help= 'path of the PyTorch model')
    parser.add_argument('--small', help= 'Model size to choose | True: 96x64 & False: 128x128')
    parser.add_argument('--input_dir', help= 'directory with your images')
    parser.add_argument('--output_dir', help= 'optical flow images will be stored here as .npy files')
    args= parser.parse_args()
    main(args)
