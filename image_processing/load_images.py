
import os

import pickle
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_processing import processing


def findFiles(path: str):
    valid_images = []
    images = os.listdir(path)
    for i in images:
        # if "_" not in i and "-"not in i:
        valid_images.append(i.lower())
    return sorted(valid_images)


def clean_hero(img: np.array):
    # convert to RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Find the contours
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # contours, hierarchy = cv2.findContours(
    #     edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get the actual inner list of hierarchy descriptions
    hierarchy = hierarchy[0]

    for index, component in enumerate(zip(contours, hierarchy)):
        currentContour = component[0]
        currentHierarchy = component[1]
        x, y, w, h = cv2.boundingRect(currentContour)
        if currentHierarchy[2] < 0:
            # these are the innermost child components
            pass
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        elif currentHierarchy[3] < 0:
            # these are the outermost parent components
            canvas = img.copy()

            arclen = cv2.arcLength(currentContour, True)
            eps = 0.0005
            epsilon = arclen * eps
            approx = cv2.approxPolyDP(currentContour, epsilon, True)
            for pt in approx:
                cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)
                cv2.drawContours(canvas, [approx], -1,
                                 (0, 0, 255), 2, cv2.LINE_AA)
            plt.imshow(canvas)
            plt.show()


if __name__ == "__main__":
    # files = findFiles("*.png")
    # output = glob.glob("hero*.png")
    # print(output)
    # for i in output:
    #     image = cv2.imread(i)
    #     clean_hero(image)
    #     raise KeyboardInterrupt

    image = cv2.imread("./test_ss.png")

    original = image.copy()
    heroes = processing.getHeroes(original)

    # print(files)
