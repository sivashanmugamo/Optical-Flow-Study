'''
Author: @sivashanmugamo
'''

import cv2 as cv
import numpy as np

cap = cv.VideoCapture("OF/data/test.mp4")
# cap= cv.VideoCapture('OF/data/japan_crossing.mp4')
ret, first_frame= cap.read()
prev_gray= cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
mask= np.zeros_like(first_frame)
mask[..., 1]= 255

while(cap.isOpened()):
    ret, frame = cap.read()
    cv.imshow('Input Video Feed', frame)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow_vector = cv.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv.cartToPolar(flow_vector[..., 0], flow_vector[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    cv.imshow('Dense Optical Flow', rgb)
    prev_gray = gray_frame
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()