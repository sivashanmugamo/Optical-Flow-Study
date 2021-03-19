'''
Author: @sivashanmugamo
'''

# Importing required libraries
import cv2 as cv
import numpy as np

# Setting paramter pointers for features & LK
feature_parameters= dict(maxCorners= 300, qualityLevel= 0.2, minDistance= 2, blockSize= 7)
lk_parameters= dict(winSize= (15,15), maxLevel= 2, criteria= (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Setting video capture to read the video file
cap= cv.VideoCapture("OF/data/test.mp4")
# cap= cv.VideoCapture('OF/data/japan_crossing.mp4')

color= (0, 255, 0)
ret, first_frame= cap.read()
print(first_frame.shape)
prev_gray_frame= cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
prev_frame= cv.goodFeaturesToTrack(prev_gray_frame, mask = None, **feature_parameters)
# cv.imshow('test', prev_frame)
print(prev_frame.shape)
mask= np.zeros_like(first_frame)

# while(cap.isOpened()):
#     ret, current_frame= cap.read()
#     cv.imshow('Input Video Feed', current_frame)
#     gray_frame= cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
#     next, status, error = cv.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, prev_frame, None, **lk_parameters)
#     good_old= prev_frame[status == 1]
#     good_new= next[status == 1]
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b= new.ravel()
#         c, d= old.ravel()
#         mask= cv.line(mask, (a, b), (c, d), color, 2)
#         current_frame= cv.circle(current_frame, (a, b), 3, color, -1)
#     output= cv.add(current_frame, mask)
#     prev_gray_frame= gray_frame.copy()
#     prev_frame= good_new.reshape(-1, 1, 2)
#     cv.imshow("Sparse OF", output)
    
#     if cv.waitKey(10) & 0xFF == ord('q'):
#         break

cap.release()
cv.destroyAllWindows()