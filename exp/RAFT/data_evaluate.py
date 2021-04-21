'''
Author: @sivashanmugamo

To run: "/Users/shiva/opt/anaconda3/envs/ML Environment/bin/python" data_evaluate.py --dir /Users/shiva/Documents/GitHub/OF/exp/RAFT/output_imgs
'''

# Importing required libraries
import os, argparse
import numpy as np

class OFEvaluation:
    def __init__(self, output_dir: str, truth_dir: str) -> None:
        
        self.output_dir= output_dir
        self.truth_dir= truth_dir

        self.result= {
            'aae': None, 
            'aepe': None, 
            'ame': None, 
            'name': None, 
            'eng': None, 
            'rms': None, 
            'nie': None
        }

        self.load_data(path= self.output_dir)
        
    # Average Angular Error
    def AAE(self):
        pass

    # Average Endpoint Error(AEPE)
    def AEPE(self):
        pass

    # Average Magnitude Error
    def AME(self):
        pass

    # Normalized Average Magnitude Error
    def NAME(self):
        pass

    # Error Normal to the Gradient (???)

    # Root-Mean-Square Error
    def RMS(self):
        pass

    # Normalized Interpolation Error
    def NIE(self):
        pass

    def evaluate(self):
        self.AEPE()
        pass

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