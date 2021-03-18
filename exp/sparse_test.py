'''
Author: @sivashanmugamo
'''

import cv2 as cv
import numpy as np

feature_parameters= dict(maxCorners= 300, qualityLevel= 0.2, minDistance= 2, blockSize= 7)
filter_parameters= dict(winSize= (15,15), maxLevel= 2, criteria= (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cap= cv.VideoCapture("OF/data/test.mp4")
# cap= cv.VideoCapture('OF/data/japan_crossing.mp4')
color= (0, 255, 0)
ret, first_frame= cap.read()
prev_gray_frame= cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
prev_frame= cv.goodFeaturesToTrack(prev_gray_frame, mask = None, **feature_parameters)
mask= np.zeros_like(first_frame)

while(cap.isOpened()):
    ret, frame= cap.read()
    cv.imshow('Input Video Feed', frame)
    gray_frame= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, prev_frame, None, **filter_parameters)
    good_old= prev_frame[status == 1]
    good_new= next[status == 1]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b= new.ravel()
        c, d= old.ravel()
        mask= cv.line(mask, (a, b), (c, d), color, 2)
        frame= cv.circle(frame, (a, b), 3, color, -1)
    output= cv.add(frame, mask)
    prev_gray_frame= gray_frame.copy()
    prev_frame= good_new.reshape(-1, 1, 2)
    cv.imshow("Sparse OF", output)
    
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()