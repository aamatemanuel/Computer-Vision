import cv2
import numpy as np

'''
##############################################################################
INITIALIZE VIDEO PARAMETERS
##############################################################################
'''

# PROJECT: ASSIGNMENT 1 - FROM BASIC IMAGE PROCESSING TOWARDS OBJECT DETECTION
# HECTOR MANUEL ARTEAGA AMATE --> r0819325

source = 'Video_CV.mp4'
# VIDEO CAPTURE OBJECT
video_cap = cv2.VideoCapture(source)
height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))

# RESIZE THE OUTPUT FRAME
w_out = int(width*0.8)
h_out = int(height*0.8)

# CONFIG FOR OUTPUT VIDEO
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Assignment1_Arteaga_Hector.mp4', fourcc, fps, (w_out, h_out))
win_name = 'Video Preview'
cv2.namedWindow(win_name)

# IMAGE TO BE USED IN TEMPLATE MATCHING
# TEMPLATE MATCHING IS MAINLY USED IN GRAY SCALE
template = cv2.imread('Wax_template.png', 0)
h, w = template.shape

# IMAGE TO BE USED IN CARTE BLANCHE
ball = cv2.imread('Volleyball.png', cv2.IMREAD_UNCHANGED)

'''
##############################################################################
DEFINITION OF FUNCTIONS
1. Print text in certain frame.
2. Gaussian Blur
3. Bilateral Filter
4. Morphological Transformations: Dilation and Erosion.
5. Sobel Operator 
6. Hough Transform --> Circles
7. Template Matching
##############################################################################
'''

# SUBTITLES
def subtitle(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_sub = cv2.putText(frame, text, (340, 40), font, 1, (255, 0, 0), 3)
    return img_sub

# FILTERING
def gaussian_blur(frame, kernel, sigmaX):
    gblur = cv2.GaussianBlur(frame, kernel, sigmaX)
    return gblur

def bilateral_blur(frame, sigma, filter_size):
    sigma_color = sigma
    sigma_space = sigma
    bblur = cv2.bilateralFilter(frame, filter_size, sigma_color, sigma_space)
    return bblur

# DILATION - MORPHOLOGICAL TRANSFORMATIONS
def dilation(img, selector):
    # Convert to HSV SCALE
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # CREATE A BLUE IMAGE TO BE USED TO SHOW THE EFFECT OF THE
    # MORPHOLOGICAL TRANSFORMATION
    bkg = np.zeros((img.shape[0], img.shape[1], 3))
    bkg[:] = 255, 0, 0
    # Lower and Upper boundaries for detection of colors in HSV format
    # Color = RED
    l1 = np.array([170, 130, 185])
    u1 = np.array([180, 255, 255])
    # Create a mask for defined color
    mask1 = cv2.inRange(hsv, l1, u1)
    # APPLY MORPHOLOGICAL TRANSFORMATION (MT)
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask1, kernel, iterations=1)
    # CALCULATE THE EFFECT OF THE MT
    diff = dil - mask1

    if selector == 0:
        img_det = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
    if selector == 1:
        img_det = cv2.bitwise_and(img, img, mask=dil)  # mask1
    if selector == 2:
        # SHOW MT IN DIFFERENT COLOR
        # Calculate the dilation effect and show it in blue color
        mt_eff = cv2.bitwise_and(bkg, bkg, mask=diff)
        img_det = cv2.bitwise_and(img, img, mask=mask1)
        dil_2 = img_det + mt_eff
        img_det = dil_2.astype(np.uint8)
    return img_det

def sobel_operator(img, dx, dy):
    bkg = np.zeros((img.shape[0], img.shape[1], 3))
    bkg[:] = 0, 255, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    sobel = cv2.Sobel(img2, cv2.CV_64F, dx, dy, ksize=5)
    cv2.normalize(sobel, dst=sobel, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    # SOBEL IS COMPUTED AS A FLOAT NUMBER
    # First is normalized, then converted to integer between 0 and 255
    sobel_int = (255 * sobel).astype(np.uint8)

    # VISUALIZE IN COLOR
    # The algorithm used was similar to the one used to show
    # the effect of the morphological transformation. The edges
    # detected by sobel are mainly black or white, so a mask is
    # created and then show this mask in green.
    l1 = np.array([160, 160, 160])
    u1 = np.array([255, 255, 255])
    l2 = np.array([0, 0, 0])
    u2 = np.array([90, 90, 90])
    mask1 = cv2.inRange(sobel_int, l1, u1)
    mask2 = cv2.inRange(sobel_int, l2, u2)
    mask3 = mask1 + mask2
    res = cv2.bitwise_and(bkg, bkg, mask=mask3)
    res = res.astype(np.uint8)
    return res

def hough_circles(img, param):
    img2 = img
    # Convert to gray-scale
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    img_blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    # Apply hough transform on the image
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 0.8, param[0], param1=param[1], param2=param[2], minRadius=param[3], maxRadius=param[4])
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle --> Green
            cv2.circle(img2, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle --> Red
            cv2.circle(img2, (i[0], i[1]), 2, (0, 0, 255), 3)
    return img2

def template_matching(img, template):
    h, w = template.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    location = min_loc
    location_1 = (location[0] - 5, location[1] - 5)
    bottom_right = (location[0] + w + 5, location[1] + h + 5)
    # CONVERTING TO INTEGER
    result = (255 * result).astype(np.uint8)
    result = cv2.bitwise_not(result)
    # Computing the likelihood map, the image experiments a change of size, due to the
    # matching process. A black background was created to have always the same image size.
    background = np.zeros_like(gray)
    h, w = result.shape
    starting_h = int((height - h) / 2)
    starting_w = int((width - w) / 2)
    background[starting_h:starting_h + h, starting_w:starting_w + w] = result

    frame = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    return frame, location_1, bottom_right

def change_object(frame, ball):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l1 = np.array([20, 155, 85])
    u1 = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, l1, u1)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(mask, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Skip if contour is small (Avoid noisy contours)
        if cv2.contourArea(contour) < 400:
            continue
        # Get the box boundaries
        (x, y, w, h) = cv2.boundingRect(contour)
        # Compute size of the box
        size = (h + w) // 2
        if y + size < height and x + size < width:
            # Resize volleyball
            volley = cv2.resize(ball, (size + 16, size + 16))
            b, g, r, a = cv2.split(volley)
            volley = cv2.merge((b, g, r))
            # Create a mask of the ball
            img2gray = cv2.cvtColor(volley, cv2.COLOR_BGR2GRAY)
            # Create a mask through the alpha channel of the png picture
            _, volley_mask = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)
            # Region of interest (ROI), where the ball will be inserted
            roi = frame[y - 8:y + size + 8, x - 8:x + size + 8]
            # Mask out logo region and insert the new one
            roi[np.where(volley_mask)] = 0
            roi += volley
    img = frame
    return img


'''
##############################################################################
MAIN
##############################################################################
'''

# WHILE LOOP TO READ THE DIFFERENT FRAMES
while (video_cap.isOpened()):

    has_frame, frame = video_cap.read()
    time = int(video_cap.get(cv2.CAP_PROP_POS_MSEC))
    if not has_frame:
        break

    # SECTION 2.1 --> CHANGE BETWEEN SPACE COLORS
    if time < 4000:
        if time < 1000 or (time >= 3000 and time < 4000):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            text = 'Space color = GRAY MODE'
        if time >= 1000 and time < 2000:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            text = 'Space color = HSV MODE'
        if time >= 2000 and time < 3000:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text = 'Space color = RGB MODE'

    # SECTION 2.2 --> FILTERING AND BLURRING
    # GAUSSIAN FILTER
    if (time >= 4000 and time < 8000):
        sigmaX = 0
        if time < 6000:
            kernel_size = (7, 7)
        else:
            kernel_size = (21, 21)
        frame = gaussian_blur(frame, kernel_size, sigmaX)
        text = 'GAUSSIAN FILTER, Kernel = ' + str(kernel_size[0]) + '   SigmaX =' + str(sigmaX)

    # BILATERAL FILTER
    # SIGMA COLOR AND SIGMA SPACE WILL REMAIN CONSTANT = 75
    if (time >= 8000 and time < 12000):
        sigma = 75
        if time < 10000:
            filter_size = 9
        else:
            filter_size = 15
        frame = bilateral_blur(frame, sigma, filter_size)
        # text = 'BILATERAL FILTER, Sigma = ' + str(sigma) + '   Filter size =' + str(filter_size)
        text = 'BILATERAL FILTER: shows clearer the edges but is slower'

    # SECTION 2.3 --> COLOR DETECTION + DILATION
    if (time >= 12000 and time < 20000):
        if time < 14000:
            selector = 0
            text = 'MASK for color detection --> RED'
        if time >= 14000 and time <17000:
            selector = 1
            text = 'Color detection WITH Dilation'
        if time >= 17000:
            selector = 2
            text = 'Dilation EFFECT --> Fills holes and complete edges'
        frame = dilation(frame,selector)

    # SECTION 3.1 --> SOBEL OPERATOR FOR EDGE DETECTION
    if (time >= 20000 and time < 25000):
        # STEP 1: USING GAUSSIAN FILTER TO ELIMINATE NOISE
        kernel_size = (9,9)
        sigmaX = 0
        frame = gaussian_blur(frame, kernel_size, sigmaX)
        if time < 22000:
            dx = 1
            dy = 0
            frame = sobel_operator(frame, dx, dy)
            text = 'SobelX -> Vertical Lines'
        elif time >= 22000 and time < 24000:
            dx = 0
            dy = 1
            frame = sobel_operator(frame, dx, dy)
            text = 'SobelY -> Horizontal Lines'
        else:
            dx = 1
            dy = 0
            frame1 = sobel_operator(frame, dx, dy)
            dx = 0
            dy = 1
            frame2 = sobel_operator(frame, dx, dy)
            frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
            text = 'Sobel --> Horizontal + Vertical lines'

    # SECTION 3.2 --> HOUGH-TRANSFORM
    if (time >= 25000 and time < 37000):
        if time < 27500:
            text = 'HOUGH TRANSFORM'
            param = np.array([80, 80, 45, 15, 110])
        if (time >= 27500 and time < 30000):
            text = 'Low canny threshold --> More wrong circles'
            param = np.array([80, 10, 45, 15, 110])
        if (time >= 30000 and time < 32500):
            text = 'High minRADIUS --> No detection of coins'
            param = np.array([80, 80, 45, 60, 110])
        if (time >= 32500 and time < 35000):
            text = 'Smaller voting threshold --> More false circles'
            param = np.array([80, 80, 10, 15, 110])
        if (time > 35000):
            text = 'Low minimum distance between circles --> ):'
            param = np.array([15, 80, 45, 15, 110])
        frame = hough_circles(frame, param)

    # SECTION 3.3 --> TEMPLATE-MATCHING
    if time >= 37000 and time < 43000:
        matched, location_1, bottom_right = template_matching(frame, template)
        if time < 39000:
            if time % 2 == 0:
                cv2.rectangle(frame, location_1, bottom_right, (0, 0, 255), 4)
            else:
                frame = frame
            text = 'Template matching'
        else:
            frame = matched
            text = 'Likelihood map - METHOD: MEAN SQUARED ERROR'

    # SECTION 4 --> CARTE BLANCHE
    if time >= 43000:
        text2 = ' '
        if time >= 44000 and time < 45500:
            text2 = '1. COLOR DETECTION'
        if time >= 45500 and time < 48000:
            text2 = '2. DILATION'
        if time >= 48000 and time < 50500:
            text2 = '3. FINDING CONTOURS ON THE MASK'
        if time >= 50500 and time < 53000:
            text2 = '4. OBTAIN BOUNDING RECTANGLE'
        if time >= 53000 and time < 55500:
            text2 = '5. RESIZE IMAGE TO BE PLACED'
        if time >= 55000 and time < 58000:
            text2 = 'Additional: Use of alpha channel'
        text = 'CARTE BLANCHE: ' + text2
        frame = change_object(frame, ball)
        if time >=60000:
            break
    frame = subtitle(frame, text)
    cv2.imshow(win_name, frame)
    b = cv2.resize(frame, (w_out, h_out), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    out.write(b)
    key = cv2.waitKey(1)

    if key == ord('Q') or key == ord('q') or key == 27:
        break

video_cap.release()
out.release()
cv2.destroyWindow(win_name)