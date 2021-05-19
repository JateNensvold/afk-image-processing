import matplotlib.pyplot as plt

# from skimage.feature import canny
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
import numpy as np
import cv2


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


def blur_image(image: np.ndarray) -> np.ndarray:
    """
    """
    v = np.median(image)
    sigma = 0.33

    # ---- apply optimal Canny edge detection using the computed median----
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    neighborhood_size = 7
    blurred = cv2.GaussianBlur(
        image, (neighborhood_size, neighborhood_size), 0)
    canny = cv2.Canny(blurred, lower_thresh, upper_thresh, 1)
    # print("Gaussian")
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
    # canny = cv2.Canny(blurred, lower_thresh, upper_thresh, 1)
    # print("bilateral")
    # plt.imshow(canny)
    # plt.show()

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    # plt.imshow(dilate)
    # plt.show()
    return canny


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
    # contours, hierarchy = cv2.findContours(
    #     binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get the actual inner list of hierarchy descriptions
    hierarchy = hierarchy[0]
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
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        elif currentHierarchy[3] < 0:
            # these are the outermost parent components

            arclen = cv2.arcLength(currentContour, True)
            eps = 0.0005
            epsilon = arclen * eps
            bestapprox = cv2.approxPolyDP(currentContour, epsilon, True)

            for pt in bestapprox:
                # dot_x = pt[0][0]
                # dot_y = pt[0][1]
                # print("dot", dot_x, dot_y, gray[dot_x][dot_y])
                cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)
                cv2.drawContours(canvas, [bestapprox], -1,
                                 (0, 0, 255), 2, cv2.LINE_AA)

        # Convert image to BGRA so it has an alpha channel

        # Draw the countours onto the mask
        cv2.drawContours(mask, contours, index, (255,)*img.shape[2], -1)

        # Combine mask and img to replace contours of original image with
        #   transparent background
        out = cv2.bitwise_and(img, mask)
    # print("center:", [i/2 for i in image.shape])
    # plt.imshow(canvas)
    # plt.show()
    return out, bestapprox


def getHeroes(image: np.array, sizeAllowanceBoundary: int = 0.15):
    """
    Parse a screenshot or image of an AFK arena hero roster into sub
        components that represent all the heroes in the image

    Args:
        image: image/screenshot of hero roster
        sizeAllowanceBoundary: percentage that each 'contour' boundary must be
            within the average contour size
    Return:
        dictionary of subimages that represent all heroes in original 'image'
    """
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # canny = cv2.Canny(blurred, 120, 255, 1)

    # kernel = np.ones((5, 5), np.uint8)
    # dilate = cv2.dilate(canny, kernel, iterations=1)
    dilate = blur_image(gray)

    # plt.imshow(cv2.cvtColor(dilate, cv2.COLOR_BGR2RGB))

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
        # print("diff: {} tolerance: {} {} {}".format(diff, tolerance, h, w))

        # ROI = original[y:y+h, x:x+w]

        if diff < tolerance and (h*w) > 2500:

            # print(c)
            # print(type(c), c.shape)
            # print(type(c[0]), c[0].shape)
            # print(type(c[0][0]), c[0][0].shape)
            # print(c[0][0])
            x = x
            a = x+w
            y = y
            b = y + h

            outside = []

            outside.append([x, y])
            # outside.append([[x2, y1]])
            # outside.append([[x1, y2]])
            outside.append([a, b])
            addStatus = True
            for shape in customContour:
                x1, y1 = shape[0]
                a1, b1 = shape[1]
                if not (a < x1 or a1 < x or b < y1 or b1 < y):
                    addStatus = False
                    break
            print("addStatus", addStatus)

            if not addStatus:
                continue
            print("added")

            if h*w not in sizes:
                sizes[h*w] = []
            sizes[h*w].append((h, w, x, y))

            customContour.append(outside)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            # cv2.drawContours(image, c, -1, 255, 3)
            # Temp
            # ------------------------------------------------------
            # arclen = cv2.arcLength(c, True)
            # eps = 0.0005
            # epsilon = arclen * eps
            # bestapprox = cv2.approxPolyDP(c, epsilon, True)

            # for pt in bestapprox:
            #     # dot_x = pt[0][0]
            #     # dot_y = pt[0][1]
            #     # print("dot", dot_x, dot_y, gray[dot_x][dot_y])
            #     cv2.circle(image, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)
            #     cv2.drawContours(image, [bestapprox], -1,
            #                      (0, 0, 255), 2, cv2.LINE_AA)
            # # ------------------------------------------------------

            # print(h, w)
            heights.append(h)
            widths.append(w)
            # ROI = original[y:y+h, x:x+w]
            # plt.figure()
            # plt.imshow(image)
            # plt.figure()

        image_number += 1

    print()

    mean = np.mean([i for i in sizes.keys()])
    h_mean = np.median(heights)
    w_mean = np.median(widths)

    print("mean: ", mean)
    lowerBoundary = 1.0 - sizeAllowanceBoundary
    upperBoundary = 1.0 + sizeAllowanceBoundary
    h_low = h_mean * lowerBoundary
    h_high = h_mean * upperBoundary
    w_low = w_mean * lowerBoundary
    w_high = w_mean * upperBoundary

    occurrences = 0
    valid_sizes = {}
    for size, dimensions in sizes.items():
        print(dimensions)
        for i in dimensions:
            if (h_low <= i[0] <= h_high) and \
                    (w_low <= i[1] <= w_high):
                occurrences += 1
                if size not in valid_sizes:
                    valid_sizes[size] = []
                valid_sizes[size].append(i)
            else:
                print(h_low, i[0], h_high, w_low, i[1], w_high)
                print(i)

    length = 0
    for i in sizes.values():
        length += len(i)
    print("occurrences: {}/{} {}%".format(occurrences,
          length, occurrences/length))
    print("sizes", sorted(valid_sizes))
    heroes = {}
    for size, dimensions in valid_sizes.items():
        for d in dimensions:
            ROI = original[d[3]:
                           d[3]+d[0],
                           d[2]:
                           d[2]+d[1]]
            # plt.imshow(ROI)
            # plt.show()
            out, poly = remove_background(ROI)
            name = "hero_{}x{}_{}x{}.png".format(d[3], d[2], d[0], d[1])
            heroes[name] = {}
            heroes[name]["image"] = out
            heroes[name]["poly"] = poly
            heroes[name]["dimensions"] = {}
            heroes[name]["dimensions"]["y"] = (d[3], d[3]+d[0])
            heroes[name]["dimensions"]["x"] = (d[2], d[2]+d[1])

    return heroes


if __name__ == "__main__":

    import os
    files = os.listdir("..")
    print(files)
    image = cv2.imread("../test_ss.png")

    heroes = getHeroes(image)
    for name, image in heroes.items():
        plt.imshow(image[0])
        plt.show()
        cv2.imwrite(name, image[0])
