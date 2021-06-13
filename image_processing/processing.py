
# from skimage.feature import canny
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
import numpy as np
import cv2
import os

import image_processing.build_db as BD
import image_processing.globals as GV
import image_processing.processing as pr
import image_processing.load_images as load
import image_processing.stamina as stamina

from typing import Sequence


def grayscale(rgb_image: str) -> np.ndarray:
    """
    Turns rgb images into grayscale images

    Args:
        rgb_image: np.ndarray of rgb elements representing an image

    Returns:
        an numpy.ndarray that is the same size as 'rgb_image' but with the
            channel dimension removed
    """
    return rgb2gray(rgba2rgb(rgb_image))


def load_image(image_path: str) -> np.ndarray:
    """
    Loads image from image path
    Args:
        rgb_image: np.ndarray of rgb elements representing an image

    Returns:
        numpy.ndarray of rgb elements
    """
    return io.imread(image_path)


def blur_image(image: np.ndarray, dilate=False,
               hsv_range: Sequence[np.array] = None) -> np.ndarray:
    """
    Applies Gaussian Blurring or HSV thresholding to image in an attempt to
        reduce noise in the image. Additionally dilation can be applied to
        further reduce image noise when the dilation parameter is true

    Args:
        image: BGR image
        dilate: flag to dilate the image after applying noise reduction.
        hsv_range: Sequence of 2 numpy arrays that represent the (lower, upper)
            bounds of the HSV range to threshold on. If this argument is None
            or False a gaussian blur will be used instead
        
    Returns:
        Image with gaussianBlur/threshold applied to it
    """
    if hsv_range:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        output = cv2.inRange(image, hsv_range[0], hsv_range[1])

    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        v = np.median(image)
        sigma = 0.33

        # ---- apply optimal Canny edge detection using the computed median----
        # automated
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))

        # preset
        # lower_thresh = (hMin = 0 , sMin = 0, vMin = 0)
    # (hMin = 0 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 197)

        neighborhood_size = 7
        blurred = cv2.GaussianBlur(
            image, (neighborhood_size, neighborhood_size), 0)
        output = cv2.Canny(blurred, lower_thresh, upper_thresh, 1)
    # print("Gaussian")
    # plt.figure()
    # plt.imshow(canny)
    # plt.show()

    # blurred = cv2.blur(image, ksize=(neighborhood_size, neighborhood_size))
    # canny = cv2.Canny(blurred, lower_thresh, upper_thresh, 1)
    # print("blur")
    # plt.imshow(canny)
    # plt.show()

    # sigmaColor = sigmaSpace = 75.
    # blurred = cv2.bilateralFilter(
    #     image, neighborhood_size, sigmaColor, sigmaSpace)
    # output = cv2.Canny(blurred, lower_thresh, upper_thresh, 1)
    # print("bilateral")
    # plt.imshow(canny)
    # plt.show()

    # plt.figure()
    # plt.imshow(dilate)
    # plt.show()
    if dilate:
        kernel = np.ones((1, 1), np.uint8)
        return cv2.dilate(output, kernel, iterations=1)

    return output


def remove_background(img):
    """
    Remove all pixels from each hero that are outside their headshot and
        replace with a transparent alpha layer

    Args
        img: a headshot of a hero

    Return:
        A tuple of (hero, poly) where hero is a headshot of the hero with a
            transparent background, and poly are the points tracing the
            headshot outline
    """
    canvas = img.copy()
    # convert to RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find the contours

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get the actual inner list of hierarchy descriptions
    hierarchy = hierarchy[0]

    # Convert image to BGRA so it has an alpha channel
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # Create a mask with all the same channels as img
    mask = np.zeros_like(img)
    for index, component in enumerate(zip(contours, hierarchy)):
        currentContour = component[0]
        currentHierarchy = component[1]
        x, y, w, h = cv2.boundingRect(currentContour)
        if currentHierarchy[2] < 0:
            # these are the innermost child components
            pass
        elif currentHierarchy[3] < 0:
            # these are the outermost parent components

            arclen = cv2.arcLength(currentContour, True)
            eps = 0.0005
            epsilon = arclen * eps
            bestapprox = cv2.approxPolyDP(currentContour, epsilon, True)

            for pt in bestapprox:
                cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)
                cv2.drawContours(canvas, [bestapprox], -1,
                                 (0, 0, 255), 2, cv2.LINE_AA)

        # Draw the countours onto the mask
        cv2.drawContours(mask, contours, index, (255,)*img.shape[2], -1)

        # Combine mask and img to replace contours of original image with
        #   transparent background
        out = cv2.bitwise_and(img, mask)
    return out, bestapprox


def getHeroContours(image: np.array, sizeAllowanceBoundary, **blurKwargs):
    """
    Args:
        image: hero roster screenshot

    Return:
        dict with imageSize as key and coordinates(h, w, x, y) as values
    """

    dilate = blur_image(image, **blurKwargs)

    if GV.DEBUG:
        load.display_image(dilate, display=True)

    # Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate through contours and filter for ROI
    image_number = 0
    sizes = {}
    heights = []
    widths = []
    customContour = []
    for c in cnts:

        x, y, w, h = cv2.boundingRect(c)
        diff = abs(h-w)
        avg_h_w = ((h+w)/2)
        tolerance = avg_h_w * 0.2

        if GV.DEBUG:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        if diff < tolerance and (h*w) > 2500:

            a = x + w
            b = y + h

            outside = []

            outside.append([x, y])
            outside.append([a, b])
            addStatus = True
            for shape in customContour:
                x1, y1 = shape[0]
                a1, b1 = shape[1]
                if not (a < x1 or a1 < x or b < y1 or b1 < y):
                    addStatus = False
                    break
            if not addStatus:
                continue

            if h*w not in sizes:
                sizes[h*w] = []
            sizes[h*w].append((h, w, x, y))

            customContour.append(outside)
            heights.append(h)
            widths.append(w)

        image_number += 1
    if GV.DEBUG:
        load.display_image(image, display=True)

    # mean = np.mean([i for i in sizes.keys()])
    h_mean = np.median(heights)
    w_mean = np.median(widths)

    # print("mean: ", mean)
    lowerBoundary = 1.0 - sizeAllowanceBoundary
    upperBoundary = 1.0 + sizeAllowanceBoundary
    h_low = h_mean * lowerBoundary
    h_high = h_mean * upperBoundary
    w_low = w_mean * lowerBoundary
    w_high = w_mean * upperBoundary

    occurrences = 0
    valid_sizes = {}
    for size, dimensions in sizes.items():
        for i in dimensions:
            if (h_low <= i[0] <= h_high) and \
                    (w_low <= i[1] <= w_high):
                occurrences += 1
                if size not in valid_sizes:
                    valid_sizes[size] = []
                valid_sizes[size].append(i)

    length = len(cnts)
    # for i in sizes.values():
    #     length += len(i)
    print("occurrences: {}/{} {}%".format(occurrences,
          length, occurrences/length * 100))
    # print("sizes", sorted(valid_sizes))
    return valid_sizes


def getHeroes(image: np.array, sizeAllowanceBoundary: int = 0.25,
              maxHeroes: bool = True,
              hsv_range: bool = False,
              removeBG: bool = False,
              si_adjustment: int = 0.1,
              row_elim: int = 3):
    """
    Parse a screenshot or image of an AFK arena hero roster into sub
        components that represent all the heroes in the image

    Args:
        image: image/screenshot of hero roster
        sizeAllowanceBoundary: percentage that each 'contour' boundary must be
            within the average contour size
        maxHeroes: flag that tells the function to experiment with multiple
            preprocessing algorithms to find the maximum number of valid heroes
        removeBG: flag to attempt to remove the background from each hero
            returned
        si_adjustment: percent of the image dimensions to take on the left and
            top side of the image to ensure si30/40 encapsulation during hero
            detection
        row_elim: minimum row size to allow (helps eliminate false positives
            that are near the same size as the median hero detection)
    Return:
        dictionary of subimages that represent all heroes from the passed in
            'image'
    """
    original = image.copy()
    heroes = {}
    valid_sizes = {}
    baseArgs = (image, sizeAllowanceBoundary)
    if maxHeroes:
        multiValid = []
        # multiValid.append(getHeroContours(*baseArgs, dilate=True))

        multiValid.append(getHeroContours(
            *baseArgs, hsv_range=hsv_range, dilate=True))

        # multiValid.append(getHeroContours(*baseArgs))

        multiValid = sorted(multiValid, key=lambda x: len(x), reverse=True)
        valid_sizes = multiValid[0]
    else:
        valid_sizes = getHeroContours(*baseArgs)

    for size, dimensions in valid_sizes.items():
        for d in dimensions:
            # h, w, x, y
            x = d[2]
            w = d[1]
            y = d[3]
            h = d[0]
            if si_adjustment:
                x_adjust = int(w * si_adjustment)
                y_adjust = int(h * si_adjustment)
                x2 = x+w
                y2 = y+h
                x = max(0, x-x_adjust)
                y = max(0, y-y_adjust)
            ROI = original[y:
                           y2,
                           x:
                           x2]

            # staminaLib.signatureItemFeatures(ROI)

            name = "hero_{}x{}_{}x{}.png".format(d[3], d[2], d[0], d[1])
            cv2.imwrite("./tempHero/{}".format(name), ROI)

            heroes[name] = {}

            if removeBG:

                out, poly = remove_background(ROI)
                heroes[name]["image"] = out
                heroes[name]["poly"] = poly
            else:
                heroes[name]["image"] = ROI

            heroes[name]["dimensions"] = {}
            heroes[name]["dimensions"]["y"] = (d[3], d[3]+d[0])
            heroes[name]["dimensions"]["x"] = (d[2], d[2]+d[1])
    rows = stamina.generate_rows(heroes)
    row_delete = []
    for index in range(len(rows)):
        if len(rows[index]) < row_elim:
            row_delete.append(index)
    for row_num in row_delete[::-1]:
        for heroTuple in rows[row_num]:
            name = heroTuple[1]
            del heroes[name]
        del rows[row_num]

    return heroes, rows


if __name__ == "__main__":

    imageDB = BD.buildDB(enrichedDB=True)
    siPath = GV.siPath

    siTempPath = os.path.join(siPath, "temp")

    for imagePath in os.listdir(siTempPath):
        rosterImage = cv2.imread(os.path.join(GV.siPath, "temp", imagePath))
        heroes = pr.getHeroes(rosterImage)
        # cropHeroes = load.crop_heroes(heroes)

        for name, imageDict in heroes.items():
            heroImage = imageDict["image"]

            heroLabel, _ = imageDB.search(heroImage)
            # import matplotlib.pyplot as plt
            # plt.imshow(heroImage)
            # plt.show()
