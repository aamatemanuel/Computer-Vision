# import the opencv library
import cv2

# define a video capture object
vid1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
num_img = 1
if not vid1.isOpened() or not vid2.isOpened():
    print("Error opening video")

while (vid1.isOpened() and vid2.isOpened()):

    # Capture the video frame by frame
    ret1, frame1 = vid1.read()
    ret2, frame2 = vid2.read()

    key = cv2.waitKey(1) & 0xFF
    if ret1 and ret2:

        # Display the resulting frame
        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)

        if key == ord('s'):
            path = "./StereoL/"
            file_name = path + "calib-{:04d}".format(num_img) + ".png"
            cv2.imwrite(file_name, frame1)

            path = "./StereoR/"
            file_name = path + "calib-{:04d}".format(num_img) + ".png"
            cv2.imwrite(file_name, frame2)

            num_img += 1

    if key == ord('q'):
        break

# After the loop release the cap object
vid1.release()
vid2.release()
# Destroy all the windows
cv2.destroyAllWindows()