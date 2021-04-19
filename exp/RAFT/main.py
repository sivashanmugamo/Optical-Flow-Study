'''
Author: @sivashanmugamo

python main.py --model --input --output
'''

# Importing required libraries
import os, argparse
import numpy as np
from PIL import Image

import torch
from torch import nn, autograd, optim
from torch.cuda import is_available
from torch.nn import functional as F

from core.raft import RAFT
from core.utils.utils import InputPadder
from pipelines import eval

DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_images(path: str) -> dict:
    '''
    '''
    
    for r, _, f in os.walk(path):
        img_list= [load_image(os.path.join(r, each_img)) for each_img in sorted(f)]

    grouped_img_list= [img_list[n:n+2] for n in range(0, len(img_list), 2)]

    return grouped_img_list

def load_image(path: str) -> torch.tensor:
    '''
    '''

    img= np.array(Image.open(path)).astype(np.int8)

    # Np to tensor & tensor rearrangement (HWC -> CHW) to support PyTorch
    img= torch.from_numpy(img).permute(2, 0, 1).float()

    return torch.unsqueeze(img, dim= 0)

def main(args):
    '''
    '''

    imgs= get_images(path= args.input)

    model= torch.nn.DataParallel(RAFT(args))

    loaded_model= torch.load(args.model, map_location= DEVICE)

    model.load_state_dict(loaded_model)
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for (frame_1, frame_2) in imgs:
            padder= InputPadder(frame_1.shape)
            frame_1, frame_2= padder.pad(frame_1, frame_2)
            break

if __name__ == '__main__':
    '''
    '''
    parser= argparse.ArgumentParser()

    parser.add_argument('--model')
    parser.add_argument('--small')
    parser.add_argument('--input')
    parser.add_argument('--output')

    args= parser.parse_args()
    main(args)