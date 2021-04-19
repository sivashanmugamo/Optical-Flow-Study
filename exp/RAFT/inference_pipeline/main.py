'''
Author: @sivashanmugamo

To run (for OF inference):
python main.py --model exp/RAFT/models/raft-kitti.pth --input exp/RAFT/input_imgs --output exp/RAFT/output_imgs
'''

import argparse

from inference_pipeline.data import *

def main(args):
    print(args)

if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--model', help= 'Path of the model')
    parser.add_argument('--small', help= 'Size of the model')
    parser.add_argument('--input', help= 'Input image directory')
    parser.add_argument('--output', help= 'Output storage directory')
    args= parser.parse_args()
    main(args)