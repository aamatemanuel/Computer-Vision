"""
MAIN.PY is the function that should be ran to start the program. It initialises and runs the threads for the menu and camera
"""
import threading
import time
from parameters import *
from camerafunctions import *

# Wait for the camera
time.sleep(4)

def run_camera_script(parameters):
    """
    This function is called as a seperate thread. It shares the parameters data structure with the 'menu' thread
    It does the following things:
        - print the camera name
        - initialise a camera lock
        - initialise the camera variables
        - call the infinite camera loop
    """
    realsense_ctx = detect_camera()
    frame_mutex = initialise_lock()
    pipe, cfg, pose_sensor, profile = connect_camera(frame_mutex, parameters)
    camera_loop(pipe, frame_mutex, pose_sensor, parameters)


if __name__ == "__main__":
    # initialise menu
    parameters = parameters_obj()
    parameters.add_param()

    # create camera thread
    camera_thread = threading.Thread(target=run_camera_script, args=(parameters,))
    camera_thread.start()
    print('started camera thread')

    # append camera thread to the threads list, the menu is run in the main loop.
    threads = list()
    threads.append(camera_thread)

    # join all the threads
    for index, thread in enumerate(threads):
        # I think we're supposed to join the threads, but for some reason that breaks the program
        print('joining thread ' + str(index))
        # thread.join()

    # start the menu
    root = menu_box(parameters.parameters_lst)
    root.mainloop()
    print('closed menu')
