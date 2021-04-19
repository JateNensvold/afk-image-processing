
import os

import pickle
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import image_processing

import image_processing.processing as processing


def findFiles(path: str, flag=False, lower=False):
    valid_images = []
    images = glob.glob(path)
    # images = os.listdir(path)
    for i in images:
        if flag or ("_" not in i and "-"not in i):
            if lower:
                i = i.lower()
            valid_images.append(i)
    return sorted(valid_images)


def clean_hero(img: np.array, lowerb: int, upperb: int):
    copy = img.copy()
    print(lowerb, upperb)
    # convert to RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # create a binary thresholded image
    # _, binary = cv2.threshold(
    #     gray, 240, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    # refines all edges in the binary image
    # kernel = np.ones((5, 5), np.uint8)  # square image kernel used for erosion
    binary = cv2.inRange(gray, lowerb, upperb, 255)
    # erosion = cv2.erode(binary, kernel, iterations=1)
    # opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    # this is for further removing small noises and holes in the image
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(closing)
    # plt.show()
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
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI = copy[y:y+h, x:x+w]
            # plt.imshow(ROI)
            # plt.show()
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
    plt.imshow(img)
    plt.show()


def colorClassify(img: np.ndarray, lower, upper):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 208, 94], dtype="uint8")
    upper = np.array([179, 255, 232], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    # Find contours
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract contours depending on OpenCV version
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate through contours and filter by the number of vertices
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        if len(approx) > 5:
            cv2.drawContours(img, [c], -1, (36, 255, 12), -1)

    plt.imshow(mask)
    plt.show()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # files = findFiles("*.png")
    # output = glob.glob("hero*.png")
    # print(output)
    # for i in output:
    #     image = cv2.imread(i)
    #     clean_hero(image)
    #     raise KeyboardInterrupt
    print(os.listdir(".."))
    name = "../lorsan.png"
    image = cv2.imread(name)

    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histr = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    # print(histr)
    zipped = list(enumerate(histr))
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    largest = (0, 0)
    colorRange = 10
    for i in range(25, len(zipped)-colorRange):
        color = zipped[i][0]
        count = 0
        for j in range(i, i+colorRange):
            count += zipped[i][1]
        if count > largest[0]:
            largest = (count, color)
    print(largest)
    # hist, edges = np.histogram(image, bins=10)
    # print(hist)
    # print(edges)
    # tesT_image = image.copy()
    # colorClassify(image, largest[0], largest[0]+10)
    clean_hero(image, largest[1], largest[1]+10)
    # gray = cv2.cvtColor(tesT_image, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite("./lorsan_gray.png", gray)

    # original = image.copy()
    # heroes = processing.getHeroes(original)

    # print(files)
