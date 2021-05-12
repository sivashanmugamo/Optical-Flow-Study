'''
Author: @sivashanmugamo

To run: "/Users/shiva/opt/anaconda3/envs/ML Environment/bin/python" data_evaluate.py --dir /Users/shiva/Documents/GitHub/OF/exp/RAFT/output_imgs
'''

# Importing required libraries
import os, argparse
import numpy as np
import torch

class OFEvaluation:
    def __init__(self, output_dir: str, truth_dir: str) -> None:
        
        self.output_dir= output_dir
        self.truth_dir= truth_dir

        self.result= {
            'aae': None, 
            'aepe': None, 
            'ame': None, 
            'name': None
        }

        self.load_data(path= self.output_dir)
        
    # Average Angular Error
    def AAE(self, truth: tuple, pred: tuple):
        ut, vt= truth
        ue, ve= pred

        M, N= ut.shape

        ut, vt, ue, ve= torch.from_numpy(ut), torch.from_numpy(vt), torch.from_numpy(ue), torch.from_numpy(ve)

        num= (ut * ue) + (vt * ve) + 1
        den= torch.sqrt((torch.square(ut) + torch.square(vt) + 1)*(torch.square(ue) + torch.square(ve) + 1))
        rhs= torch.arccos(torch.div(num, den))

        return sum(rhs)/(M*N)

    # Average Endpoint Error(AEPE)
    def AEPE(self, truth: tuple, pred: tuple):
        ut, vt= truth
        ue, ve= pred

        M, N= ut.shape

        rhs= np.sqrt(np.square(ue-ut) + np.square(ve-vt))

        return np.sum(rhs)/(M*N)

    # Average Magnitude Error
    def AME(self, truth: tuple, pred: tuple):
        ut, vt= truth
        ue, ve= pred

        M, N= ut.shape

        rhs= np.abs(np.sqrt(np.square(ue) + np.square(ve)) - np.sqrt(np.square(ut) + np.square(vt)))

        return np.sum(rhs)/(M*N)

    # Normalized Average Magnitude Error
    def NAME(self, truth: tuple, pred: tuple):
        ut, vt= truth
        ue, ve= pred

        M, N= ut.shape

        num= np.abs(np.sqrt(np.square(ue) + np.square(ve)) - np.sqrt(np.square(ut) + np.square(vt)))
        den= np.sqrt(np.square(ut) + np.square(vt))
        rhs= np.divide(num, den)

        return np.sum(rhs)/(M*N)

    def evaluate(self, truth: tuple, pred: tuple):
        self.AAE()
        self.AEPE()
        self.AME()
        self.NAME()

    def load_data(self, path: str) -> None:
        '''
        '''

        data= list()
        for r, d, f in os.walk(path):
            for each in f:
                data.append(np.load(os.path.join(r, each)))

        self.ofs= data.copy()

    def save_result(self):
        pass

def main(args):
    obj= OFEvaluation(
        output_dir= args.output, 
        truth_dir= args.truth
    )

if __name__ == '__main__':

    parser= argparse.ArgumentParser()
    parser.add_argument('--output', help= '')
    parser.add_argument('--truth', help= '')
    args= parser.parse_args()

    main(args)