'''
Creates a numpy file with paired frame data

Author: @sivashanmugamo
'''

# Importing required libraries
import cv2
import numpy as np

from datetime import datetime

class framePairing:
    def __init__(self, vid_path) -> None:
        '''
        '''

        self.vid_path= vid_path

    def pairing(self):
        '''
        '''
        start_time= datetime.now()

        vid_data= cv2.VideoCapture(self.vid_path)

        self.frame_np= list()
        prev_frame= None

        i= 0
        while vid_data.isOpened():
            temp_np= list()

            ret, frame= vid_data.read()
            
            if type(frame) is np.ndarray:
                curr_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if ret:
                    if type(prev_frame) is np.ndarray:
                        temp_np.append(prev_frame)
                        temp_np.append(curr_frame)
                        self.frame_np.append(temp_np)
                        prev_frame= curr_frame
                    else:
                        prev_frame= curr_frame
            else:
                raise TypeError('NumPy Array expected, but received {}.'.format(type(frame)))

            if ret==False:
                print('This should have been here')
                break

        self.frame_np= np.array(self.frame_np)
        end_time= datetime.now()

    def save(self, path):
        np.save('frame_test.npy', self.frame_np)

    def load(self, path):
        with open(path, 'rb') as f:
            return np.load(f)

# -----------------------------------------------------------------------------
# GETTER SETTER METHODS
# -----------------------------------------------------------------------------

    @property
    def vid_path(self):
        return self._vid_path

    @vid_path.setter
    def vid_path(self, value):
        if isinstance(value, str) == False:
            raise TypeError('String expected for video file path, but received {}.'.format(type(value)))
        self._vid_path= value
