# %%
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


def remove_background(img):
    canvas = img.copy()
    # convert to RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

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

            arclen = cv2.arcLength(currentContour, True)
            eps = 0.0005
            epsilon = arclen * eps
            approx = cv2.approxPolyDP(currentContour, epsilon, True)
            gray_range = [0, 255]
            approx
            for pt in approx:
                # dot_x = pt[0][0]
                # dot_y = pt[0][1]
                # print("dot", dot_x, dot_y, gray[dot_x][dot_y])
                cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)
                cv2.drawContours(canvas, [approx], -1,
                                 (0, 0, 255), 2, cv2.LINE_AA)
            plt.imshow(canvas)
            plt.show()

            # Convert image to RGBA so it has an alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            # Create a mask with all the same channels as img
            mask = np.zeros_like(img)

            # Draw the countours onto the mask
            cv2.drawContours(mask, contours, index, (255,)*img.shape[2], -1)

            # Combine mask and img to replace contours of original image with
            #   transparent background
            out = cv2.bitwise_and(img, mask)
    print("center:", [i/2 for i in image.shape])
    plt.imshow(canvas)
    plt.show()
    plt.imshow(out)
    plt.show()
    return out


def getHeroes(image: np.array):
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    canny = cv2.Canny(blurred, 120, 255, 1)

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)

    print("dilate")
    plt.imshow(cv2.cvtColor(dilate, cv2.COLOR_BGR2RGB))
    plt.show()
    # Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("cnts1", len(cnts))
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    print("cnts2", len(cnts))

    # Iterate through contours and filter for ROI
    image_number = 0
    sizes = {}
    heights = []
    widths = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        diff = abs(h-w)
        avg_h_w = ((h+w)/2)
        tolerance = avg_h_w * 0.2
        print("diff: {} tolerance: {} {} {}".format(diff, tolerance, h, w))
        if diff < tolerance:
            if h*w not in sizes:
                sizes[h*w] = []
            sizes[h*w].append((h, w, x, y))
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            print(h, w)
            heights.append(h)
            widths.append(w)
            ROI = original[y:y+h, x:x+w]

        image_number += 1

    mean = np.mean([i for i in sizes.keys()])
    h_mean = np.median(heights)
    w_mean = np.median(widths)

    print("mean: ", mean)
    h_low = h_mean * .9
    h_high = h_mean * 1.1
    w_low = w_mean * .9
    w_high = w_mean * 1.1

    occurances = 0
    valid_sizes = {}
    for size, dimensions in sizes.items():
        for i in dimensions:
            print(h_low, i[0], h_high, w_low, i[1], w_high)
            if (h_low <= i[0] <= h_high) and \
                    (w_low <= i[1] <= w_high):
                occurances += 1
                if size not in valid_sizes:
                    valid_sizes[size] = []
                valid_sizes[size].append(i)
    length = 0
    for i in sizes.values():
        length += len(i)
    print("occurances: {}/{} {}%".format(occurances,
          length, occurances/length))
    print(sorted(valid_sizes))
    heroes = {}
    for size, dimensions in valid_sizes.items():
        for d in dimensions:
            ROI = original[d[3]:
                           d[3]+d[0],
                           d[2]:
                           d[2]+d[1]]

            # image_to_write = cv2.cvtColor(ROI, cv2.COLOR_RGB2BGR)
            # plt.imshow(image_to_write)
            # plt.show()

            # cv2.imwrite("ROI_{}_{}_{}x{}.png".format(d[3], d[2], d[0], d[1]),
            #             image_to_write)
            # plt.imshow(ROI)
            # plt.show()

            out = remove_background(ROI)
            name = "hero_{}x{}_{}x{}.png".format(d[3], d[2], d[0], d[1])
            # raise KeyboardInterrupt
            heroes[name] = out
    return heroes


if __name__ == "__main__":

    import os
    files = os.listdir("..")
    print(files)
    image = cv2.imread("../test_ss.png")

    heroes = getHeroes(image)
    import os
    for name, image in heroes.items():

        cv2.imwrite(name, image)


# %%
