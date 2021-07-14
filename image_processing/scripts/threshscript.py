import cv2
# import sys
import numpy as np
import argparse


def nothing(x):
    pass


def threshold(image):
    # Load in image
    # image = cv2.imread(
    #     "/home/nate/projects/afk-image-processing/test_2.jpg")

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
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while(1):

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')

        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) |
           (psMax != sMax) | (pvMax != vMax)):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d,"
                  " vMax = %d)" % (
                      hMin, sMin, vMin, hMax, sMax, vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image', output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def color_threshold(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.window
    # create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('RMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('GMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('BMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('RMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('GMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('BMax', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('RMax', 'image', 255)
    cv2.setTrackbarPos('GMax', 'image', 255)
    cv2.setTrackbarPos('BMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    RMin = GMin = BMin = RMax = GMax = BMax = 0
    pRMin = pGMin = pBMin = pRMax = pGMax = pBMax = 0

    output = image
    wait_time = 33

    while(1):

        # get current positions of all trackbars
        RMin = cv2.getTrackbarPos('RMin', 'image')
        GMin = cv2.getTrackbarPos('GMin', 'image')
        BMin = cv2.getTrackbarPos('BMin', 'image')

        RMax = cv2.getTrackbarPos('RMax', 'image')
        GMax = cv2.getTrackbarPos('GMax', 'image')
        BMax = cv2.getTrackbarPos('BMax', 'image')

        # Set minimum and max HSV values to display
        lower = np.array([RMin, GMin, BMin])
        upper = np.array([RMax, GMax, BMax])

        # Create HSV Image and threshold into a range.
        RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(RGB, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((pRMin != RMin) | (pGMin != GMin) | (pBMin != BMin) | (pRMax != RMax) |
           (pGMax != GMax) | (pBMax != BMax)):
            print("(RMin = %d , GMin = %d, BMin = %d), (RMax = %d , GMax = %d,"
                  " BMax = %d)" % (
                      RMin, GMin, BMin, RMax, GMax, BMax))
            pRMin = RMin
            pGMin = GMin
            pBMin = BMin
            pRMax = RMax
            pGMax = GMax
            pBMax = BMax

        # Display output image
        cv2.imshow('image', output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = cv2.imread(
        "/home/nate/downloads/Screenshot_20210625-203702_AFK_Arena.jpg")

    parser = argparse.ArgumentParser(
        description='Threshold determination script.')
    parser.add_argument(
        "-c", "--COLOR", help="Runs the program in RGB mode",
        action="store_true")
    parser.add_argument(
        "-i", "--IMAGE", help="Image to load")

    args = parser.parse_args()

    COLOR = args.COLOR
    IMAGE = args.IMAGE

    if IMAGE:
        image = cv2.imread(IMAGE)

    if COLOR:
        print("color")
        multiplier = 4
        image = cv2.resize(
            image, (image.shape[1]*multiplier, image.shape[0]*multiplier))
        color_threshold(image)
    else:
        threshold(image)
