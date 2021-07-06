
# from skimage.feature import canny
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
import numpy as np
import cv2
import os

import rtree

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
               hsv_range: Sequence[np.array] = None,
               rgb_range: Sequence[np.array] = None,
               reverse: bool = False) -> np.ndarray:
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
        rgb_range: Sequence of 2 numpy arrays that represent the (lower, upper)
            bounds of the RGB range to threshold on. If this argument is None
            or False a gaussian blur will be used instead
        reverse: flag to bitwise_not the image after applying hsv_range

    Returns:
        Image with gaussianBlur/threshold applied to it
    """

    # # (hMin = 0 , sMin = 0, vMin = 150), (hMax = 179 , sMax = 255, vMax = 255)
    # lower_hsv_filter = np.array([0, 121, 79])
    # upper_hsv_filter = np.array([16, 252, 175])
    # # (hMin = 0 , sMin = 121, vMin = 79), (hMax = 16 , sMax = 252, vMax = 175)

    # hvs_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # output = cv2.inRange(hvs_image, lower_hsv_filter, upper_hsv_filter)
    # # output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # _, hsv_mask = cv2.threshold(output, 3, 255, cv2.THRESH_BINARY)
    # # cv2.COLOR_HSV2
    # # h, s, v1 = cv2.split(output)

    # mask_inv = cv2.bitwise_not(output)

    if hsv_range:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        output = cv2.inRange(image, hsv_range[0], hsv_range[1])
        if reverse:
            output = cv2.bitwise_not(output)
        # output = cv2.bitwise_and(output, mask_inv)
    elif rgb_range:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output = cv2.inRange(image, rgb_range[0], rgb_range[1])
        if reverse:
            output = cv2.bitwise_not(output)
        # output = cv2.bitwise_and(output, mask_inv)
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
    # ("Gaussian")

    # blurred = cv2.blur(image, ksize=(neighborhood_size, neighborhood_size))
    # canny = cv2.Canny(blurred, lower_thresh, upper_thresh, 1)
    # ("blur")

    # sigmaColor = sigmaSpace = 75.
    # blurred = cv2.bilateralFilter(
    #     image, neighborhood_size, sigmaColor, sigmaSpace)
    # output = cv2.Canny(blurred, lower_thresh, upper_thresh, 1)
    # ("bilateral")

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


def getHeroContours(image: np.array, sizeAllowanceBoundary, display=None, **blurKwargs):
    """
    Args:
        image: hero roster screenshot

    Return:
        dict with imageSize as key and coordinates(h, w, x, y) as values
    """

    dilate = blur_image(image, **blurKwargs)

    if GV.DEBUG:
        load.display_image(dilate)

    # Find contours
    import imutils
    import imutils.contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = imutils.grab_contours(cnts)
    # cnts = imutils.contours.sort_contours(cnts,
    #                                       method="left-to-right")[0]
    cnts = sorted(cnts, key=cv2.contourArea,
                  reverse=True)

    # Iterate through contours and filter for ROI
    image_number = 0
    sizes = {}
    heights = []
    widths = []
    customContour = []

    idx = rtree.index.Index()

    for _index, c in enumerate(cnts):

        x, y, w, h = cv2.boundingRect(c)
        diff = abs(h-w)
        avg_h_w = ((h+w)/2)
        tolerance = avg_h_w * 0.2

        # if GV.DEBUG and display:
        _idx_coords = (x, y, x+w, y+h)
        _intersections = list(idx.intersection(_idx_coords))
        if _intersections:
            for _collision in _intersections:
                # print(_collision)
                pass
        else:
            idx.insert(_index, _idx_coords)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # print("cnt", x, y)
            # ROI = image[y:
            #             y+h,
            #             x:
            #             x+w]
            # cv2.drawContours(image, [c], -1, (0, 0, 255), thickness=2)

            # load.display_image(image, display=True)
            # cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

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
            # sizes[h*w].append((h, w, x, y))
            sizes[h*w].append((x, y, w, h))

            customContour.append(outside)
            heights.append(h)
            widths.append(w)

        image_number += 1

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

        for _coords in dimensions:
            if (h_low <= _coords[3] <= h_high) or \
                    (w_low <= _coords[2] <= w_high):
                occurrences += 1
                # if size not in valid_sizes:
                #     valid_sizes[size] = []
                name = "{}x{}_{}x{}".format(
                    _coords[0], _coords[1], _coords[2], _coords[3])

                valid_sizes[name] = _coords

    length = len(cnts)
    # for i in sizes.values():
    #     length += len(i)
    print("occurrences: {}/{} {}%".format(occurrences,
          length, occurrences/length * 100))
    # print("sizes", sorted(valid_sizes))
    return valid_sizes


def getHeroes(image: np.array, sizeAllowanceBoundary: int = 0.25,
              maxHeroes: bool = True,
              #   hsv_range: bool = False,
              removeBG: bool = False,
              si_adjustment: int = 0.2,
              row_elim: int = 3,
              blur_args: dict = {}):
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
            top side of the image to ensure si30/40 capture during hero
            detection(False/None for no adjustment)
        row_elim: minimum row size to allow (helps eliminate false positives
            i.e. similar size shapes as median hero shape) that are near the
            same size as the median hero detection)
        blur_args: keyword arguments for `processing.blur_image` method

    Return:
        dictionary of subimages that represent all heroes from the passed in
            'image'
    """
    original_modifiable = image.copy()
    original_unmodifiable = image.copy()

    heroes = {}
    valid_sizes = {}
    baseArgs = (original_modifiable, sizeAllowanceBoundary)
    multi_valid = []

    # if maxHeroes:
    # multi_valid.append(getHeroContours(*baseArgs, dilate=True))
    blur_args["hsv_range"] = [
        np.array([0, 0, 0]), np.array([179, 255, 192])]
    multi_valid.append(getHeroContours(
        *baseArgs, **blur_args))

    # (hMin = 0 , sMin = 0, vMin = 74), (hMax = 27 , sMax = 253, vMax = 255)
    baseArgs = (image.copy(), sizeAllowanceBoundary)
    blur_args["hsv_range"] = [
        np.array([0, 0, 74]), np.array([27, 253, 255])]
    blur_args["reverse"] = True
    multi_valid.append(getHeroContours(
        *baseArgs, display=True, **blur_args))

    # baseArgs = (image.copy(), sizeAllowanceBoundary)
    # blur_args["hsv_range"] = [
    #     np.array([5, 79, 211]), np.array([21, 106, 250])]
    # blur_args["reverse"] = True
    # multi_valid.append(getHeroContours(
    #     *baseArgs, **blur_args))

    # multi_valid.append(getHeroContours(*baseArgs))

    # multi_valid = sorted(multi_valid, key=lambda x: len(x), reverse=True)
    # valid_sizes = multi_valid[0]

#     valid_sizes = getHeroContours(*baseArgs)

    import statistics
    hero_widths = []
    hero_heights = []

    for _heroes_list in multi_valid:
        for _object_name, _object_dimensions in _heroes_list.items():
            hero_widths.append(_object_dimensions[2])
            hero_heights.append(_object_dimensions[3])
    hero_widths.sort()
    hero_heights.sort()

    hero_w_median = statistics.median(hero_widths)
    hero_h_median = statistics.median(hero_heights)

    spacing = round((hero_w_median + hero_h_median)/10)
    print("spacing", spacing)
    hero_matrix = stamina.matrix(spacing=spacing)
    for _hero_list in multi_valid:
        for _object_name, _object_dimensions in _hero_list.items():
            hero_matrix.auto_append(_object_dimensions, _object_name)
    # hero_matrix.prune(threshold=5)
    hero_matrix.sort()

    for _row_index, _row in enumerate(hero_matrix):
        print("row({}) length: {}".format(_row_index, len(_row)))
        for _object_index, _object in enumerate(_row):

            x = _object[0][0]
            y = _object[0][1]
            w = _object[0][2]
            h = _object[0][3]

            y2 = y + h
            x2 = x + w

            _hero_name = _object[1]
            # print("name", _object)

            # x2 = x+w
            # y2 = y+h
            # x = max(0, x-x_adjust)
            # y = max(0, y-y_adjust)
            ROI = original_unmodifiable[y:
                                        y2,
                                        x:
                                        x2]
            if si_adjustment:
                x_adjust = round(w * si_adjustment)
                y_adjust = round(h * si_adjustment)

                _new_x = max(round(x - x_adjust), 0)
                _new_y = max(round(y - y_adjust), 0)
                new_ROI = original_unmodifiable[_new_y:
                                                y2,
                                                _new_x:
                                                x2]
                new_ROI = new_ROI.copy()
                # lyca # (hMin = 4 , sMin = 69, vMin = 83), (hMax = 23 , sMax = 255, vMax = 255)
                blurred = blur_image(new_ROI, reverse=True, hsv_range=[
                    np.array([4, 69, 83]), np.array([23, 255, 355])])

                new_contours = cv2.findContours(
                    blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                import imutils
                new_contours = imutils.grab_contours(new_contours)
                new_contours = sorted(new_contours, key=cv2.contourArea,
                                      reverse=True)[0]
                # save_input = input("Press any key to continue, or enter s to save this image")
                # if save_input == "s":
                #     import image_processing.scripts.threshscript as thresh
                #     thresh.threshold(new_ROI)
                new_x, new_y, new_w, new_h = cv2.boundingRect(new_contours)
                # cv2.rectangle(new_ROI, (new_x, new_y),
                #               (new_x+new_w, new_y+new_h), (0, 0, 255), 2)

                new_contours = [new_contours]

                cv2.fillPoly(new_ROI, new_contours, [255, 0, 0])
                # (dimensions, name)
                output = _row.check_collision(
                    ((x2-new_w, y2-new_h, new_w, new_h), _hero_name))

                x = _row[_object_index][0][0]
                y = _row[_object_index][0][1]
                w = _row[_object_index][0][2]
                h = _row[_object_index][0][3]

                y2 = y + h
                x2 = x + w
                w_border_offset = max(round(0.03 * w), 2)
                h_border_offset = max(round(0.03 * h), 2)

                ROI = original_unmodifiable[y - h_border_offset:
                                            y2,
                                            x - w_border_offset:
                                            x2]

            # staminaLib.signatureItemFeatures(ROI)
            # cv2.imwrite("./tempHero/{}".format(_hero_name), ROI)

            heroes[_hero_name] = {}

            if removeBG:

                out, poly = remove_background(ROI)

                heroes[_hero_name]["image"] = out
                heroes[_hero_name]["poly"] = poly
            else:
                heroes[_hero_name]["image"] = ROI

            heroes[_hero_name]["object"] = _object
            # heroes[_hero_name]["dimensions"] = {}
            # heroes[_hero_name]["dimensions"]["y"] = (d[3], d[3]+d[0])
            # heroes[_hero_name]["dimensions"]["x"] = (d[2], d[2]+d[1])
        # load.display_image([heroes[_hero[1]]["image"]
        #                     for _hero in _row], multiple=True, display=True)
    # row_delete = []
    # for index in range(len(rows)):
    #     if len(rows[index]) < row_elim:
    #         row_delete.append(index)
    # for row_num in row_delete[::-1]:
    #     for heroTuple in rows[row_num]:
    #         name = heroTuple[1]
    #         del heroes[name]
    #     del rows[row_num]

    # for _row in hero_matrix:
    #     load.display_image([heroes[_hero[1]]["image"]
    #                        for _hero in _row], multiple=True, display=True)

        # for _hero in _row:
        #     _name = _hero[1]
        #     load.display_image(heroes[_name]["image"])

    return heroes, hero_matrix


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
