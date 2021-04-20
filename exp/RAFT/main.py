'''
Recurrent All-Pairs Field Transforms (RAFT)
Author: @sivashanmugamo

Extracts per-pixel features; builds multi-scale 4D correlation volumes for all pairs of pixels;
and iteratively updates a flow field through a recurrent unit that performs lookups on the 
correlation volumes.

python main.py --model --input --output --small --mixed_precision
'''

# Importing required libraries
import os, time, argparse
import numpy as np
from PIL import Image

import torch
from torch import nn, autograd, optim
from torch.cuda import is_available
from torch.nn import functional as F

# Importing functions from the Princeton's RAFT repo
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_images(path: str) -> list:
    '''
    Returns all image data from a given path

    Input:
        path: str - Path of the input images directory
    Returns:
        list - List of all image data 
    '''
    
    for r, _, f in os.walk(path):
        img_list= [load_image(os.path.join(r, each_img)) for each_img in sorted(f)]

    grouped_img_list= [img_list[n:n+2] for n in range(0, len(img_list), 2)]

    return grouped_img_list

def load_image(path: str) -> torch.tensor:
    '''
    Returns image data with PyTorch support

    Input:
        path: str - Path of the image
    Returns:
        tensor - Image data in tensor form
    '''

    img= np.array(Image.open(path)).astype(np.int8)

    # Np to tensor & tensor rearrangement (HWC -> CHW) to support PyTorch
    img= torch.from_numpy(img).permute(2, 0, 1).float()

    return torch.unsqueeze(img, dim= 0)

def save_images(path: str, imgs: dict) -> None:
    '''
    Saves infered optical flow data in specified directory

    Input:
        path: str - Path to directory where the images should be saved
        imgs: dict - Dictionary containing both the frames & the OF data 
                     (Scaled & unscaled)
    '''

    i= 1
    for _, val in imgs.items():
        OF= val['upscaled_OF'][0].permute(1, 2, 0).cpu().numpy()
        OF= flow_viz.flow_to_image(OF)
        OF= Image.fromarray(OF)
        OF.save(os.path.join(path, 'flow_{}.png'.format(i)))
        i+=1

def save_prediction(path: str, prediction: dict) -> None:
    '''
    Saves the infered optical data as a .npy file for evaluation

    Input:
        path: str - Path to directory where the prediction data should be saved
        prediction: dict - Dictionary containing both the frames & the OF data 
                           (Scaled & unscaled)
    '''

    i= 1
    for _, val in prediction.items():
        pred= val['upscaled_OF'].cpu().numpy()
        np.save(os.path.join(path, 'pair_{}_pred.npy'.format(i)), pred)
        i+=1

def main(args):
    '''
    Infers the Optical Flow (OF) data from the images using the specified model

    Input:
        args - Contains path to the input images, output images, model size, 
               mixed precision, & model directory
    '''

    imgs= get_images(path= args.input)

    # Initiating parallel processing support
    model= torch.nn.DataParallel(RAFT(args))

    # Loading parameters from pretrained model
    loaded_model= torch.load(args.model, map_location= DEVICE)

    model.load_state_dict(loaded_model)
    model.to(DEVICE)
    model.eval()

    i= 1
    pred_dict= dict()
    with torch.no_grad(): # Disabling gradient calculation
        for (frame_1, frame_2) in imgs:
            temp_dict= dict()
            temp_dict['frame_1']= frame_1
            temp_dict['frame_2']= frame_2

            padder= InputPadder(frame_1.shape)
            frame_1, frame_2= padder.pad(frame_1, frame_2)

            # Prediction
            unscaled, upscaled= model(frame_1, frame_2, iters= 20, test_mode= True)

            temp_dict['upscaled_OF']= upscaled
            temp_dict['unscaled_OF']= unscaled
            pred_dict['pred_{}'.format(i)]= temp_dict
            i+=1

    # Saving prediction as .npy
    save_prediction(path= args.output, prediction= pred_dict)
    
    # Saving prediction as image
    save_images(path= args.output, imgs= pred_dict)

if __name__ == '__main__':
    parser= argparse.ArgumentParser()

    parser.add_argument('--model')
    parser.add_argument('--small')
    parser.add_argument('--input')
    parser.add_argument('--mixed_precision')
    parser.add_argument('--output')

    args= parser.parse_args()
    main(args)