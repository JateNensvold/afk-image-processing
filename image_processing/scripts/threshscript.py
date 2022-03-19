import cv2
import numpy as np

import image_processing.globals as GV

GV.parser.add_argument(
    "-c", "--COLOR", help="Runs the program in RGB mode",
    action="store_true")

GV.reload_globals()

IMAGE_SIZE_MULTIPLIER = 4


def nothing(_x: None):
    """
    Stub Function used as a fake callback
    Args:
        _x (None): empty value passed to callback function
    """
    return


def threshold(image: np.ndarray):
    """
    Run a program that allows the user to set different
        Hue, Saturation and Value thresholds on the image to see how a filter
        with those values would look

    Args:
        image (np.ndarray): image to threshold
    """

    # Create a window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.window
    # create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hue_min = saturation_min = value_min = hue_max = saturation_max = \
        value_max = 0
    print_hue_min = print_saturation_min = print_value_min = print_hue_max = \
        print_saturation_max = print_value_max = 0

    output = image
    wait_time = 33

    while True:

        # get current positions of all trackbars
        hue_min = cv2.getTrackbarPos('HMin', 'image')
        saturation_min = cv2.getTrackbarPos('SMin', 'image')
        value_min = cv2.getTrackbarPos('VMin', 'image')

        hue_max = cv2.getTrackbarPos('HMax', 'image')
        saturation_max = cv2.getTrackbarPos('SMax', 'image')
        value_max = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and max HSV values to display
        lower = np.array([hue_min, saturation_min, value_min])
        upper = np.array([hue_max, saturation_max, value_max])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((print_hue_min != hue_min) |
           (print_saturation_min != saturation_min) |
           (print_value_min != value_min) | (print_hue_max != hue_max) |
           (print_saturation_max != saturation_max) |
           (print_value_max != value_max)):
            print(f"(hue_min = {hue_min} , saturation_min = {saturation_min}, "
                  f"value_min = {value_min}), (hue_max = {hue_max} , "
                  f"saturation_max = {saturation_max}, "
                  f"value_max = {value_max})")
            print_hue_min = hue_min
            print_saturation_min = saturation_min
            print_value_min = value_min
            print_hue_max = hue_max
            print_saturation_max = saturation_max
            print_value_max = value_max

        # Display output image
        cv2.imshow('image', output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def color_threshold(image: np.ndarray):
    """
    Run a program that allows the user to set different
        Red, Green and Blue thresholds on the image to see how a filter with
        those values would look

    Args:
        image (np.ndarray): image to threshold
    """
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.window
    # create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('red_min', 'image', 0, 255, nothing)
    cv2.createTrackbar('green_min', 'image', 0, 255, nothing)
    cv2.createTrackbar('blue_min', 'image', 0, 255, nothing)
    cv2.createTrackbar('red_max', 'image', 0, 255, nothing)
    cv2.createTrackbar('green_max', 'image', 0, 255, nothing)
    cv2.createTrackbar('blue_max', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('red_max', 'image', 255)
    cv2.setTrackbarPos('green_max', 'image', 255)
    cv2.setTrackbarPos('blue_max', 'image', 255)

    # Initialize to check if HSV min/max value changes
    red_min = green_min = blue_min = red_max = green_max = blue_max = 0
    print_red_min = print_green_min = print_blue_min = print_red_max = \
        print_green_max = print_blue_max = 0

    output = image
    wait_time = 33

    while True:

        # get current positions of all trackbars
        red_min = cv2.getTrackbarPos('red_min', 'image')
        green_min = cv2.getTrackbarPos('green_min', 'image')
        blue_min = cv2.getTrackbarPos('blue_min', 'image')

        red_max = cv2.getTrackbarPos('red_max', 'image')
        green_max = cv2.getTrackbarPos('green_max', 'image')
        blue_max = cv2.getTrackbarPos('blue_max', 'image')

        # Set minimum and max HSV values to display
        lower = np.array([red_min, green_min, blue_min])
        upper = np.array([red_max, green_max, blue_max])

        # Create HSV Image and threshold into a range.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(rgb_image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((print_red_min != red_min) | (print_green_min != green_min) |
                (print_blue_min != blue_min) | (print_red_max != red_max) |
                (print_green_max != green_max) | (print_blue_max != blue_max)):
            print(f"(red_min = {red_min} , green_min = {green_min}, "
                  f"blue_min = {blue_min}), (red_max = {red_max} , "
                  f"green_max = {green_max}, blue_max = {blue_max})")
            print_red_min = red_min
            print_green_min = green_min
            print_blue_min = blue_min
            print_red_max = red_max
            print_green_max = green_max
            print_blue_max = blue_max

        # Display output image
        cv2.imshow('image', output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # python3 threshscript.py <path to image>

    # if GV.ARGS.COLOR:
    #     print("Running in RGB mode")
    #     color_threshold(
    #         cv2.resize(GV.IMAGE_SS,
    #                    (GV.IMAGE_SS.shape[1]*IMAGE_SIZE_MULTIPLIER,
    #                     GV.IMAGE_SS.shape[0]*IMAGE_SIZE_MULTIPLIER)))
    # else:
    GV.global_parse_args()
    threshold(GV.IMAGE_SS)
