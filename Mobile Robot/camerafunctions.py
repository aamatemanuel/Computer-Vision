#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.
# Python 2/3 compatibility
from __future__ import print_function
import cv_functions
from Map import *

import math
import matplotlib.pyplot as plt

# Import OpenCV and numpy
import cv2
import numpy as np
from math import tan, pi

# Import Realsense Camera
import pyrealsense2 as rs

"""
This example shows how to use T265 intrinsics and extrinsics in OpenCV to
asynchronously compute depth maps from T265 fisheye images on the host.
T265 is not a depth camera and the quality of passive-only depth options will
always be limited compared to (e.g.) the D4XX series cameras. However, T265 does
have two global shutter cameras in a stereo configuration, and in this example
we show how to set up OpenCV to undistort the images and compute stereo depth
from them.
Getting started with python3, OpenCV and T265 on Ubuntu 16.04:
First, set up the virtual enviroment:
$ apt-get install python3-venv  # install python3 built in venv support
$ python3 -m venv py3librs      # create a virtual environment in pylibrs
$ source py3librs/bin/activate  # activate the venv, do this from every terminal
$ pip install opencv-python     # install opencv 4.1 in the venv
$ pip install pyrealsense2      # install librealsense python bindings
Then, for every new terminal:
$ source py3librs/bin/activate  # Activate the virtual environment
$ python3 t265_stereo.py        # Run the example
"""

def detect_camera():
    # First import the library

    # CHECK THE SERIAL NUMBER OF CAMERA
    realsense_ctx = rs.context()
    for i in range(len(realsense_ctx.devices)):
        detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
        print(detected_camera)
    return realsense_ctx


"""
In this section, we will set up the functions that will translate the camera
intrinsics and extrinsics from librealsense into parameters that can be used
with OpenCV.
The T265 uses very wide angle lenses, so the distortion is modeled using a four
parameter distortion model known as Kanalla-Brandt. OpenCV supports this
distortion model in their "fisheye" module, more details can be found here:
https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
"""

# Points clicked on the image
# https://www.tutorialspoint.com/opencv-python-how-to-display-the-coordinates-of-points-clicked-on-an-image

"""
Returns R, T transform from src to dst
"""
def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)

"""
Returns a camera matrix K from librealsense intrinsics
"""
def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                     [            0, intrinsics.fy, intrinsics.ppy],
                     [            0,             0,              1]])

"""
Returns the fisheye distortion from librealsense intrinsics
"""
def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])

# Set up a mutex to share data between threads
from threading import Lock
def initialise_lock():
    frame_mutex = Lock()
    return frame_mutex

frame_data = {"left"  : None,
                  "right" : None,
                  "timestamp_ms" : None
                  }

"""
This callback is called on a separate thread, so we must use a mutex
to ensure that data is synchronized properly. We should also be
careful not to do much work on this thread to avoid data backing up in the
callback queue.
When starting the pipeline with a callback both wait_for_frames() and 
poll_for_frames() will throw exception. Wait_for_frames() it is related with 
getting the pose. 
"""

def callback(frame, frame_mutex):
    global frame_data
    if frame.is_frameset():
        frameset = frame.as_frameset()
        # pose = frame.as_pose_frame()
        # translation = pose.get_pose_frame()
        f1 = frameset.get_fisheye_frame(1).as_video_frame()
        f2 = frameset.get_fisheye_frame(2).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        right_data = np.asanyarray(f2.get_data())
        ts = frameset.get_timestamp()
        frame_mutex.acquire()
        frame_data["left"] = left_data
        frame_data["right"] = right_data
        frame_data["timestamp_ms"] = ts
        frame_mutex.release()

def connect_camera(frame_mutex, parameters):
    # Declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs.pipeline()

    # Build config object and stream everything
    cfg = rs.config()
    cfg.enable_stream(rs.stream.fisheye, 1)
    cfg.enable_stream(rs.stream.fisheye, 2)
    cfg.enable_device('929122110612')
    device = cfg.resolve(pipe).get_device()
    pose_sensor = device.first_pose_sensor()

    # CHANGE BETWEEN AUTOEXPOSURE AND MANUAL SETTING AUTOEXPOSURE
    pose_sensor.set_option(rs.option.enable_auto_exposure, parameters.autoExposure.value)
    if not parameters.autoExposure.value:
        pose_sensor.set_option(rs.option.exposure, parameters.exposure_time.value)
        pose_sensor.set_option(rs.option.gain, parameters.exposure_gain.value)

    print(pose_sensor)
    # Start streaming with our callback
    # profile = pipe.start(cfg, callback)
    profile = pipe.start(cfg, lambda frame: callback(frame, frame_mutex))
    # left_fish = profile.get_device().query_sensors()[0]
    # print(left_fish)
    # right_fish = profile.get_device().query_sensors()[1]
    #
    # left_fish.set_option(rs.option.enable_auto_exposure, 0)
    #
    # left_fish.set_option(rs.option.exposure, 500)
    # sensor = profile.get_device().get_sensor(0)
    # sensor.set_option(rs.option.exposure, 500)
    return pipe, cfg, pose_sensor, profile

def initialise_loop(pipe, ):
    # Set up an OpenCV window to visualize the results
    WINDOW_TITLE = 'Realsense'
    # cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    # Configure the OpenCV stereo algorithm. See
    # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
    # description of the parameters
    window_size = 5
    min_disp = 0
    # must be divisible by 16
    num_disp = 112 - min_disp
    max_disp = min_disp + num_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32)

    # Retreive the stream and intrinsic properties for both cameras
    profiles = pipe.get_active_profile()
    streams = {"left": profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
               "right": profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
    intrinsics = {"left": streams["left"].get_intrinsics(),
                  "right": streams["right"].get_intrinsics()}

    # Print information about both cameras
    print("Left camera:", intrinsics["left"])
    print("Right camera:", intrinsics["right"])

    # Translate the intrinsics from librealsense into OpenCV
    K_left = camera_matrix(intrinsics["left"])
    D_left = fisheye_distortion(intrinsics["left"])
    K_right = camera_matrix(intrinsics["right"])
    D_right = fisheye_distortion(intrinsics["right"])
    (width, height) = (intrinsics["left"].width, intrinsics["left"].height)

    # Get the relative extrinsics between the left and right camera
    (R, T) = get_extrinsics(streams["left"], streams["right"])

    # We need to determine what focal length our undistorted images should have
    # in order to set up the camera matrices for initUndistortRectifyMap.  We
    # could use stereoRectify, but here we show how to derive these projection
    # matrices from the calibration and a desired height and field of view

    # We calculate the undistorted focal length:
    #
    #         h
    # -----------------
    #  \      |      /
    #    \    | f  /
    #     \   |   /
    #      \ fov /
    #        \|/
    stereo_fov_rad = 90 * (pi / 180)  # 90 degree desired fov
    stereo_height_px = 300  # 300x300 pixel stereo output
    stereo_focal_px = stereo_height_px / 2 / tan(stereo_fov_rad / 2)

    # We set the left rotation to identity and the right rotation
    # the rotation between the cameras
    R_left = np.eye(3)
    R_right = R

    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image
    stereo_width_px = stereo_height_px + max_disp
    stereo_size = (stereo_width_px, stereo_height_px)
    stereo_cx = (stereo_height_px - 1) / 2 + max_disp
    stereo_cy = (stereo_height_px - 1) / 2

    # Construct the left and right projection matrices, the only difference is
    # that the right projection matrix should have a shift along the x axis of
    # baseline*focal_length
    P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0, 0, 1, 0]])
    P_right = P_left.copy()
    P_right[0][3] = T[0] * stereo_focal_px

    # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
    # since we will crop the disparity later
    Q = np.array([[1, 0, 0, -(stereo_cx - max_disp)],
                  [0, 1, 0, -stereo_cy],
                  [0, 0, 0, stereo_focal_px],
                  [0, 0, -1 / T[0], 0]])

    # Create an undistortion map for the left and right camera which applies the
    # rectification and undoes the camera distortion. This only has to be done
    # once
    m1type = cv2.CV_32FC1
    (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
    (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
    undistort_rectify = {"left": (lm1, lm2),
                         "right": (rm1, rm2)}
    return undistort_rectify, stereo, min_disp, max_disp, num_disp, WINDOW_TITLE

def get_image(frame_mutex, undistort_rectify, stereo, min_disp, max_disp, num_disp):
    # Check if the camera has acquired any frames
    frame_mutex.acquire()
    valid = frame_data["timestamp_ms"] is not None
    frame_mutex.release()

    # If frames are ready to process
    if valid:
        # Hold the mutex only long enough to copy the stereo frames
        frame_mutex.acquire()
        frame_copy = {"left": frame_data["left"].copy(),
                      "right": frame_data["right"].copy()}
        frame_mutex.release()

        # Undistort and crop the center of the frames
        center_undistorted = {"left": cv2.remap(src=frame_copy["left"],
                                                map1=undistort_rectify["left"][0],
                                                map2=undistort_rectify["left"][1],
                                                interpolation=cv2.INTER_LINEAR),
                              "right": cv2.remap(src=frame_copy["right"],
                                                 map1=undistort_rectify["right"][0],
                                                 map2=undistort_rectify["right"][1],
                                                 interpolation=cv2.INTER_LINEAR)}

        # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
        disparity = stereo.compute(center_undistorted["left"], center_undistorted["right"]).astype(np.float32) / 16.0

        # re-crop just the valid part of the disparity
        disparity = disparity[:, max_disp:]
        # convert disparity to 0-255 and color it
        disp_vis = 255 * (disparity - min_disp) / num_disp

        # disp_color == Depth map
        # Color_image for left camera --> Just for D-Models
        disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis, 1), cv2.COLORMAP_JET)
        color_image = cv2.cvtColor(center_undistorted["left"][:, max_disp:], cv2.COLOR_GRAY2RGB)
        # cv2.imshow(WINDOW_TITLE, np.hstack((color_image, disp_color)))

        # FISH-EYE CAMERA
        fish_eye_left = frame_copy["left"]
        fish_eye_right = frame_copy["right"]
        # The size is (800,848) resize to reduce computation time.
        # Flip the images, because are inverted
        fish_l = cv2.resize(fish_eye_left, (300, 300))
        fish_r = cv2.resize(fish_eye_right, (300, 300))

        ## UNDISTORTED IMAGES
        # Gray_scale for left & right image
        left_camera = center_undistorted["left"][:, max_disp:]
        right_camera = center_undistorted["right"][:, max_disp:]
        # Flip picture
        left_camera_flip = cv2.flip(left_camera, 1)
        right_camera_flip = cv2.flip(right_camera, 1)
        return left_camera
    return None

M = None
Minv = None
IMAGE_W = None
IMAGE_H = None
def warp(left_camera_flip, parameters):
    global M, Minv, IMAGE_W, IMAGE_H

    # if firstloop: reinitialise the warping matrix
    if parameters.firstLoop.value:
        # set firstLoop back to False
        parameters.firstLoop.value = 0

        # if warpType is 1: shrink bottom method
        if parameters.warpType.value:

            M, Minv, IMAGE_W, IMAGE_H = cv_functions.initialise_birdseyeview_transform_shrink_bottom(src_x=300,
                                                                                                     src_y=300,
                                                                                                     dst_x=500,
                                                                                                     dst_y=500,initialiseWarp=parameters.initialiseWarp.value)
            print('created new M:')

        # if warpType = 0: zoom top method
        else:
            M, Minv, IMAGE_W, IMAGE_H = cv_functions.initialise_birdseyeview_transform_zoom_top(
                parameters.initialiseWarp.value, left_camera_flip)
            print('created new M:')
        print(M)

    # try to warp the image, can fail because M has not been initialised
    if type(M) != type(None):
        try:
            # Image warping & display
            warped_img = cv2.warpPerspective(left_camera_flip, M, (IMAGE_W, IMAGE_H))
            return warped_img
        except:
            return None
    return None

def camera_loop(pipe, frame_mutex, pose_sensor, parameters):
    undistort_rectify, stereo, min_disp, max_disp, num_disp, WINDOW_TITLE = initialise_loop(pipe)

    try:
        while True:
            # update camera settings
            if not parameters.autoExposure.value:
                pose_sensor.set_option(rs.option.exposure, parameters.exposure_time.value)
                pose_sensor.set_option(rs.option.gain, parameters.exposure_gain.value)

            left_camera_flip = get_image(frame_mutex, undistort_rectify, stereo, min_disp, max_disp, num_disp)
            if type(left_camera_flip) != type(None):
                cv2.imshow(WINDOW_TITLE, left_camera_flip)

                #WARPING SECTION
                if parameters.warp.value:
                    # warp the image
                    warped_img = warp(left_camera_flip, parameters)
                    if type(warped_img) != type(None): cv2.imshow('warped image', warped_img)

                # LINE DETECTION SECTION
                if parameters.useHough.value:
                    # detect the lines using a hough transform
                    if not parameters.warp.value:
                        warped_img = left_camera_flip
                    picture_with_lines, hough_lines = cv_functions.detectLines(warped_img, parameters)
                    if type(picture_with_lines) != type(None): # if it didnt fail
                        cv2.imshow('picture with lines', picture_with_lines)
                        if parameters.makeMap.value:
                            def makeMap(hough_lines, parameters):
                                myMap = Graph()
                                for line in hough_lines:
                                    myMap.addLineToGraph(line)
                                myMap.addEdgdesAsLines(xwidthtop=parameters.mapTopWidth.value, xwidthbottom=parameters.mapBottomWidth.value, xwidthtotal=picture_with_lines.shape[0],  ymax=parameters.mapHeight.value, ymin=0)
                                myMap.removeDuplicateLines()
                                myMap.splitIntersectingLines()
                                myMap.removeDuplicateLines()
                                myMap.findNeighbours()
                                myMap.findLoopsBackTrack()
                                myMap.removeDuplicateLoops()
                                myMap.displayGraphResult(warped_img.copy())
                                return myMap
                            map = makeMap(hough_lines, parameters)

                    # try to find closed contours using the lines found by the hough transform
                    # picture_with_contours = cv_functions.detect_shapes(picture_with_lines,hough_lines)
                    # if type(picture_with_contours) != type(None): cv2.imshow('picture with contours', picture_with_contours)

            # some key commands
            key = cv2.waitKey(1)
            if key == ord('s'): mode = "stack"
            if key == ord('o'): mode = "overlay"
            if key == ord('q') or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        pipe.stop()
