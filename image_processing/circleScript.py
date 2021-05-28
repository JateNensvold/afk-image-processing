import numpy as np
import argparse
import cv2
import signal

from functools import wraps
import errno
import os
import copy

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig_image = np.copy(image)
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
# cv2.waitKey(0)

circles = None

# this is the range of possible circle in pixels you want to find
# minimum_circle_size = 100
# maximum possible circle size you're willing to find in pixels
maximum_circle_size = 80

guess_dp = 1.0

number_of_circles_expected = 1  # we expect to find just one circle
breakout = False

# minimum of 1, no maximum, (max 300?) the quantity of votes
max_guess_accumulator_array_threshold = 100
# needed to qualify for a circle to be found.
circleLog = []

guess_accumulator_array_threshold = max_guess_accumulator_array_threshold

while guess_accumulator_array_threshold > 1 and breakout == False:
    # start out with smallest resolution possible, to find the most precise circle, then creep bigger if none found
    guess_dp = 1.0
    print("resetting guess_dp:" + str(guess_dp))
    while guess_dp < 9 and breakout == False:
        guess_radius = maximum_circle_size
        print("setting guess_radius: " + str(guess_radius))
        print(circles is None)
        while True:

            # HoughCircles algorithm isn't strong enough to stand on its own if you don't
            # know EXACTLY what radius the circle in the image is, (accurate to within 3 pixels)
            # If you don't know radius, you need lots of guess and check and lots of post-processing
            # verification.  Luckily HoughCircles is pretty quick so we can brute force.

            print("guessing radius: " + str(guess_radius) +
                  " and dp: " + str(guess_dp) + " vote threshold: " +
                  str(guess_accumulator_array_threshold))

            circles = cv2.HoughCircles(gray,
                                       cv2.HOUGH_GRADIENT,
                                       # resolution of accumulator array.
                                       dp=guess_dp,
                                       minDist=100,  # number of pixels center of circles should be from each other, hardcode
                                       param1=50,
                                       param2=guess_accumulator_array_threshold,
                                       # HoughCircles will look for circles at minimum this size
                                       minRadius=(guess_radius-10),
                                       # HoughCircles will look for circles at maximum this size
                                       maxRadius=(guess_radius+3)
                                       )

            if circles is not None:
                if len(circles[0]) == number_of_circles_expected:
                    print("len of circles: " + str(len(circles)))
                    circleLog.append((copy.copy(circles), {
                                     "maxRadius": guess_radius+3,
                                     "minRadius": guess_radius-10,
                                     "param2": guess_accumulator_array_threshold,
                                     "dp": guess_dp}))
                    print("k1")
                break
                circles = None
            guess_radius -= 5
            if guess_radius < 15:
                break

        guess_dp += 1.5

    guess_accumulator_array_threshold -= 2

# Return the circleLog with the highest accumulator threshold

# ensure at least some circles were found
for cirTuple in circleLog:
    # convert the (x, y) coordinates and radius of the circles to integers
    output = np.copy(orig_image)
    cir = cirTuple[0]
    if (len(cir) > 1):
        print("FAIL before")
        exit()

    print(cirTuple)

    cir = np.round(cir[0, :]).astype("int")

    for (x, y, r) in cir:
        cv2.circle(output, (x, y), r, (0, 0, 255), 2)
        cv2.rectangle(output, (x - 5, y - 5),
                      (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow("output", np.hstack([orig_image, output]))
    # input()
    cv2.waitKey(5000)
