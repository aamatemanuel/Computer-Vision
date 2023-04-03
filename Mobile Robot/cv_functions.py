import cv2
import numpy as np
import math
from sklearn.cluster import KMeans

def initialise_birdseyeview_transform_shrink_bottom(src_x, src_y, dst_x, dst_y, initialiseWarp):
    if initialiseWarp:
        left_top_x = int(input('left top x? '))
        right_top_x = int(input('right top x? '))
        left_bottom_x = int(input('left bottom x? '))
        right_bottom_x = int(input('right bottom x? '))

        src_y_max = int(input("max y value? (for the mask) "))
        src_y_min = int(input("min y value? (for the mask) "))
    else:
        left_top_x = 0
        right_top_x = 300
        left_bottom_x = 125
        right_bottom_x = 175

        src_y_max = 300
        src_y_min = 0
    left_top_src = [0, src_y_min]
    right_top_src = [src_x, src_y_min]
    left_bottom_src = [0, src_y_max]
    right_bottom_src = [src_x, src_y_max]

    bottom_middle = dst_x / 2
    # bottom_width = dst_x * (right_top_x - left_top_x) / (right_bottom_x - left_bottom_x)
    bottom_width = dst_x * (right_bottom_x - left_bottom_x) / (right_top_x - left_top_x)
    bottom_left_x = round(bottom_middle - (bottom_width) / 2)
    bottom_right_x = round(bottom_middle + (bottom_width) / 2)
    left_top_dst = [0, 0]
    right_top_dst = [dst_x, 0]
    left_bottom_dst = [bottom_left_x, dst_y]
    right_bottom_dst = [bottom_right_x, dst_y]

    src = np.float32([left_bottom_src, right_bottom_src, left_top_src, right_top_src])
    dst = np.float32([left_bottom_dst, right_bottom_dst, left_top_dst, right_top_dst])

    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation
    IMAGE_W = dst_x
    IMAGE_H = dst_y
    return M, Minv, IMAGE_W, IMAGE_H

def initialise_birdseyeview_transform_zoom_top(initialiseWarp, left_camera_flip):
    if initialiseWarp:
        # RESET THE FILE TO BE EMPTY
        file = open("warp_points.txt", "w")
        file.close()

        # CREATE A WINDOW
        cv2.namedWindow('Point Coordinates')
        cv2.setMouseCallback('Point Coordinates', click_event)
        while True:
            cv2.imshow('Point Coordinates', left_camera_flip)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        # cv2.destroyAllWindows()

    file_with_points = open("warp_points.txt", "r")
    pointstring = file_with_points.read()

    left_top = [int(pointstring[1:4]), int(pointstring[5:8])]
    right_top = [int(pointstring[12:15]), int(pointstring[16:19])]
    left_bottom = [int(pointstring[23:26]), int(pointstring[27:30])]
    right_bottom = [int(pointstring[34:37]), int(pointstring[38:41])]

    height_dst = round(max(math.sqrt((left_top[1] - left_bottom[1]) ** 2 + (left_top[0] - left_bottom[0]) ** 2),
                           math.sqrt((right_top[1] - right_bottom[1]) ** 2 + (right_top[0] - right_bottom[0]) ** 2)))
    width_dst = round(max(math.sqrt((left_top[0] - right_top[0]) ** 2 + (left_top[1] - right_top[1]) ** 2),
                          math.sqrt((left_bottom[0] - right_bottom[0]) ** 2 + (left_bottom[1] - right_bottom[1]) ** 2)))

    file_with_points.close()
    print("points for warp:")
    print(left_top)
    print(right_top)
    print(left_bottom)
    print(right_bottom)

    # https://nikolasent.github.io/opencv/2017/05/07/Bird%27s-Eye-View-Transformation.html
    # IMAGE_H = 223
    # IMAGE_W = 1280

    IMAGE_H = height_dst * 5
    IMAGE_W = width_dst * 5

    # src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    src = np.float32([left_bottom, right_bottom, left_top, right_top])
    print('SRC')
    print(src)
    # dst = np.float32([left_bottom_dst, right_bottom_dst, [0, 0], [IMAGE_W, 0]])
    # left_bottom_dst = [569, IMAGE_H]
    # right_bottom_dst = [711, IMAGE_H]
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation
    return M, Minv, IMAGE_W, IMAGE_H


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')

        # PUT COORDINATES AS TEXT IN IMAGE
        # cv2.putText(left_camera_flip, f'({x},{y})', (x, y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw point on the image
        # cv2.circle(left_camera_flip, (x, y), 3, (0, 255, 255), -1)
        file = open("warp_points.txt", "a")
        x_string = f'({x}'
        y_string = f'{y})'
        if 100 > x >= 10:
            x_string = f'(0{x}'
        if x < 10:
            x_string = f'(00{x}'
        if 100 > y >= 10:
            y_string = f'0{y})'
        if y < 10:
            y_string = f'00{y})'

        # if x > 100 and y > 100:
        #     file.write(f'({x},{y})' + " \n")
        # elif x < 100 and y > 100:
        #     file.write(f'(0{x},{y})' + " \n")
        # elif x > 100 and y < 100:
        #     file.write(f'({x},0{y})' + " \n")
        # elif x < 100 and y < 100:
        #     file.write(f'(0{x},0{y})' + " \n")
        # elif x < 10 and y > 10:
        #     file.write(f'(00{x},{y})' + " \n")
        # elif x > 10 and y < 10:
        #     file.write(f'({x},00{y})' + " \n")
        # elif x < 10 and y < 10:
        #     file.write(f'(00{x},00{y})' + " \n")

        file.write(x_string + "," + y_string + " \n")
        file.close()

def detectLines(picture, parameters):
    """
    FUNCTION TO DETECT LINES FROM THE PICTURE
    - Filtering to reduce noise, but retain edges
    - Sobel edge detector || Canny edge detector
    - Hough transform
    """

    """"########## FILTERING ##########"""
    ## Blurring:
    # blurred_image = cv2.blur(left_camera_flip,(3,3),0,)

    ## Gaussian blurring
    gaussian_blurred_image = cv2.GaussianBlur(picture, (3, 3), 0)

    ## Bilateral filtering
    # bilateral_filter_image = cv2.bilateralFilter(picture, 9, 10, 0, cv2.BORDER_ISOLATED)

    """"########## EDGE ENHANCEMENT ##########"""

    # # SOBEL OPERATOR
    # bilateral_filter_image_normalized = bilateral_filter_image / 255. # the image has to be normalized
    # sobel_x = cv2.Sobel(bilateral_filter_image_normalized, cv2.CV_64F, 1, 0, ksize=3)
    # sobel_y = cv2.Sobel(bilateral_filter_image_normalized, cv2.CV_64F, 0, 1, ksize=3)
    # cv2.normalize(sobel_x, dst=sobel_x, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    # cv2.normalize(sobel_y, dst=sobel_y, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    # sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # sobel_enhanced = cv2.sqrt(cv2.addWeighted(cv2.pow(sobel_x, 2.0), 1.0, cv2.pow(sobel_y, 2.0), 1.0, 0.0))

    ## CANNY EDGE DETECTOR
    canny = cv2.Canny(gaussian_blurred_image, parameters.cannyTresh.value, 255)

    """"########## THRESHOLDING ##########"""
    ## THRESHOLD
    thresh = parameters.edgeTreshold.value
    maxValue = 255
    retval, thresholded = cv2.threshold(gaussian_blurred_image, thresh, maxValue, cv2.THRESH_BINARY)
    # retval, sobel_threshold = cv2.threshold(sobel_enhanced, thresh, maxValue, cv2.THRESH_BINARY)
    if parameters.thresholdOrCanny.value:
        useMe = thresholded
    else:
        useMe = canny
    cv2.imshow('Threshold or Canny', useMe)

    """"########## MORPHOLOGICAL TRANSFORMATIONS ##########"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (parameters.kernelmorph.value, parameters.kernelmorph.value))
    if parameters.morphIterations.value != 0:
        if parameters.dilationOrErosion.value == 0:
            morph = cv2.dilate(useMe, kernel, iterations=parameters.morphIterations.value)
        if parameters.dilationOrErosion.value == 1:
            morph = cv2.erode(useMe, kernel, iterations=parameters.morphIterations.value)

    else:
        morph = useMe
    cv2.imshow('Morphological', morph)

    """"########## HOUGH TRANSFORM ##########"""

    dilate_color = cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB)
    hough_lines = cv2.HoughLinesP(morph,
                                  rho = 1,
                                  theta = np.pi / 180,
                                  threshold = parameters.houghthresh.value,
                                  minLineLength=parameters.minLineLength.value,
                                  maxLineGap=parameters.maxLineGap.value)

    count = 0
    if hough_lines is not None:
        for line in hough_lines:

            x1, y1, x2, y2 = line[0]
            cv2.line(dilate_color, (x1, y1), (x2, y2), (255, 255, 0), 2)
            if count == parameters.maxNbLines.value:
                ## PIECE OF CODE TO ANALYZE FRAME BY FRAME
                # list_of_lines = np.asarray(hough_lines)
                # np.save('list_of_lines.py', list_of_lines)
                # cv2.imwrite("Frame.png", dilate_color_without_lines)
                # exit()
                break
            count += 1

    return dilate_color, hough_lines

def detect_shapes(picture, hough_lines):
    grayscale = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
    bin_img = np.zeros_like(grayscale)
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(bin_img, (x1, y1), (x2, y2), 255, 2)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i = 0

    # list for storing names of shapes
    for contour in contours:

        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        arclength = cv2.arcLength(contour, True)
        threshold_arclength = 200
        if len(approx) == 4 and arclength > threshold_arclength:
            cv2.drawContours(picture, [contour], 0, (0, 0, 255), 5)
        # using drawContours() function
        #cv2.drawContours(picture, [contour], 0, (0, 0, 255), 5)
    picture2 = cv2.cvtColor(picture, cv2.COLOR_GRAY2RGB)
    return picture2



