from typing import List, Sequence, NamedTuple

import cv2
from numpy import ndarray, array, ones, uint8, median


HSV_RANGE = List[ndarray]


class HSVRange(NamedTuple):
    """
    NamedTuple with attributes used for thresholding an image
    """
    hue_min: int
    saturation_min: int
    value_min: int

    hue_max: int
    saturation_max: int
    value_max: int

    def get_range(self):
        """
        Create and Format numpy array in format passed to cv2.inrange
        """
        hsv_range: HSV_RANGE = [self.lower_range,
                                self.upper_range]
        return hsv_range

    @property
    def lower_range(self):
        """
        Create and numpy array with min values
        """
        hsv_range: ndarray = array(
            [self.hue_min, self.saturation_min, self.value_min])
        return hsv_range

    @property
    def upper_range(self):
        """
        Create and numpy array with max values
        """
        hsv_range: ndarray = array(
            [self.hue_max, self.saturation_max, self.value_max])
        return hsv_range


def blur_image(image: ndarray, dilate=False,
               hsv_range: HSVRange = None,
               rgb_range: Sequence[array] = None,
               reverse: bool = False) -> ndarray:
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

    if hsv_range:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        output = cv2.inRange(image,
                             hsv_range.lower_range,
                             hsv_range.upper_range)
        if reverse:
            output = cv2.bitwise_not(output)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        # output = cv2.bitwise_and(output, mask_inv)
    elif rgb_range:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = cv2.inRange(image, rgb_range[0], rgb_range[1])
        if reverse:
            output = cv2.bitwise_not(output)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        # output = cv2.bitwise_and(output, mask_inv)

    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_median_value = median(image)
        sigma = 0.33

        # ---- apply optimal Canny edge detection using the computed median----
        # automated
        lower_thresh = int(max(0, (1.0 - sigma) * image_median_value))
        upper_thresh = int(min(255, (1.0 + sigma) * image_median_value))

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
        kernel = ones((1, 1), uint8)
        return cv2.dilate(output, kernel, iterations=1)

    return output
