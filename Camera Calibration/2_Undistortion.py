import sys
import numpy as np
import time

import cv2

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
# 1 --> Done with st1, st2 pictures
# 2 --> Done with stereoR and stereoL pictures
# 4 --> Not using new matrix

cv_file.open('improved_params1.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode("Left_Stereo_Map_x").mat()
stereoMapL_y = cv_file.getNode("Left_Stereo_Map_y").mat()
stereoMapR_x = cv_file.getNode("Right_Stereo_Map_x").mat()
stereoMapR_y = cv_file.getNode("Right_Stereo_Map_y").mat()

cap_left = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)

while(cap_right.isOpened() and cap_left.isOpened()):
    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()
    key = cv2.waitKey(1) & 0xFF
    if succes_right and succes_left:

    ################## CALIBRATION #########################################################
        undistortedR = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        undistortedL= cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # cv2.imshow("frame_right", undistortedR)
        # cv2.imshow("frame_left", undistortedL)

        # Set stereo vision parameters
        window_size = 3
        min_disp = 0
        num_disp = 128
        block_size = window_size
        uniquenessRatio = 10
        speckleWindowSize = 100
        speckleRange = 32
        disp12MaxDiff = 1

        # Compute disparity map
        grayL = cv2.cvtColor(undistortedL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(undistortedR, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=disp12MaxDiff,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
        )
        disparity = stereo.compute(grayL, grayR)

        # Normalize and scale disparity map
        disparity = (disparity - min_disp) / num_disp
        disparity = np.uint8(disparity * 255)

        # Display disparity map
        cv2.imshow('Disparity Map', disparity)

    if key == ord('q'):
        break

# After the loop release the cap object
cap_left.release()
cap_right.release()
# Destroy all the windows
cv2.destroyAllWindows()