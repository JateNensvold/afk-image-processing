"""
Module that contains most of the Screenshot/Roster parsing capabilities

Used to split apart the raw image passed from the CLI to processable
subsections that can be fed to the Models used to detect AFK Arena Hero
Attributes
"""
import statistics

from typing import Dict, Sequence, Tuple, Union

import cv2
import rtree
import imutils
# Need this import to use imutils.contours.sort_contours,
#   without it Module raises AttributeError
# pylint: disable=unused-import
from imutils import contours  # noqa
import numpy as np

import image_processing.globals as GV
import image_processing.load_images as load
import image_processing.afk.roster.dimensions_object as DO
import image_processing.afk.roster.matrix as MA
import image_processing.afk.roster.RowItem as RI

# pylint: disable=invalid-name
HERO_INFO = Dict[str, Union[np.ndarray, RI.RowItem]]
# pylint: disable=invalid-name
HERO_DICT = Dict[str, "SegmentResult"]


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

        image_median_value = np.median(image)
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
    # create a binary image from threshold
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find the contours

    detected_contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get the actual inner list of hierarchy descriptions
    hierarchy = hierarchy[0]

    # Convert image to BGRA so it has an alpha channel
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # Create a mask with all the same channels as img
    mask = np.zeros_like(img)
    for index, component in enumerate(zip(detected_contours, hierarchy)):
        current_contour = component[0]
        current_hierarchy = component[1]
        if current_hierarchy[2] < 0:
            # these are the innermost child components
            pass
        elif current_hierarchy[3] < 0:
            # these are the outermost parent components

            arclen = cv2.arcLength(current_contour, True)
            eps = 0.0005
            epsilon = arclen * eps
            polygon_curve_approximation = cv2.approxPolyDP(
                current_contour, epsilon, True)

            for point in polygon_curve_approximation:
                cv2.circle(
                    canvas, (point[0][0], point[0][1]), 7, (0, 255, 0), -1)
                cv2.drawContours(canvas, [polygon_curve_approximation], -1,
                                 (0, 0, 255), 2, cv2.LINE_AA)

        # Draw the countours onto the mask
        cv2.drawContours(mask, detected_contours,
                         index, (255,)*img.shape[2], -1)

        # Combine mask and img to replace contours of original image with
        #   transparent background
        out = cv2.bitwise_and(img, mask)
    return out, polygon_curve_approximation


def get_hero_contours(image: np.array, size_allowance_boundary: float,
                      **blurKwargs):
    """
    Args:
        image: hero roster screenshot

    Return:
        dict with imageSize as key and
            image_processing.stamina.DimensionalObject's as values
    """

    dilate = blur_image(image, **blurKwargs)

    if GV.DEBUG:
        load.display_image(dilate)

    # Find contours
    detected_contours = cv2.findContours(
        dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = imutils.grab_contours(detected_contours)
    detected_contours = sorted(detected_contours, key=cv2.contourArea,
                               reverse=True)

    # Iterate through contours and filter for detected_hero
    image_number = 0
    sizes: dict[int, list[DO.DimensionsObject]] = {}
    heights = []
    widths = []

    idx = rtree.index.Index(interleaved=False)

    for _index, detected_contour in enumerate(detected_contours):

        image_segment_info = DO.SegmentRectangle(
            *cv2.boundingRect(detected_contour))
        dim_object = DO.DimensionsObject(image_segment_info)
        diff = abs(image_segment_info.height - image_segment_info.width)
        avg_h_w = ((image_segment_info.height + image_segment_info.width)/2)
        tolerance = avg_h_w * 0.2

        # if GV.DEBUG and display:
        idx_coords = dim_object.coords()
        intersections = list(idx.intersection(idx_coords))
        if not intersections:
            idx.insert(_index, idx_coords)
            split_coords = dim_object.coords()
            cv2.rectangle(
                image,
                split_coords.vertex1(),
                split_coords.vertex2(),
                (0, 0, 255),
                2)

        if diff < tolerance and (dim_object.size()) > 2500:

            size = dim_object.size()
            if size not in sizes:
                sizes[size] = []
            sizes[size].append(dim_object)

            heights.append(dim_object.height)
            widths.append(dim_object.width)

        image_number += 1
    load.display_image(image, display=(True and GV.DEBUG))

    h_mean = np.median(heights)
    w_mean = np.median(widths)

    lower_boundary = 1.0 - size_allowance_boundary
    upper_boundary = 1.0 + size_allowance_boundary
    h_low = h_mean * lower_boundary
    h_high = h_mean * upper_boundary
    w_low = w_mean * lower_boundary
    w_high = w_mean * upper_boundary

    occurrences = 0
    valid_sizes = {}
    for _name, size_list in sizes.items():

        for dimension_object in size_list:
            if (h_low <= dimension_object.height <= h_high) or \
                    (w_low <= dimension_object.width <= w_high):
                occurrences += 1
                segment_rectangle = dimension_object.dimensional_values()
                name = (f"{segment_rectangle.x}x{segment_rectangle.y}_"
                        f"{segment_rectangle.width}x{segment_rectangle.height}")

                valid_sizes[name] = dimension_object

    return valid_sizes


class SegmentResult:
    """
    class with info describing the location of subimage in a larger image 

        Ex. A Hero Portrait from a Hero Roster in AFK Arena
    """

    def __init__(self, segment_name: str, segment_image: np.ndarray,
                 segment_location: RI.RowItem):
        """_summary_

        Args:
            segment_name (str): _description_
            segment_image (np.ndarray): _description_
            segment_location (RI.RowItem): _description_
        """
        self.name = segment_name
        self.image = segment_image
        self.segment_location = segment_location


def get_heroes(image: np.ndarray,  blur_args: dict,
               size_allowance_boundary: int = 0.15,
               si_adjustment: int = 0.2,
               row_eliminate: int = 5,
               ) -> Tuple[HERO_DICT, MA.Matrix]:
    """
    Parse a screenshot or image of an AFK arena hero roster into sub
        components that represent all the heroes in the image

    Args:
        image: image/screenshot of hero roster
        size_allowance_boundary: percentage that each 'contour' boundary must be
            within the average contour size
        si_adjustment: percent of the image dimensions to take on the left and
            top side of the image to ensure si30/40 capture during hero
            contour re-evaluation(False/None for no adjustment)
        row_eliminate: minimum row size to allow (helps eliminate false
            positives i.e. similar size shapes as median hero shape) that are
            near the same size as the median hero detection)
        blur_args: keyword arguments for `processing.blur_image` method

    Return:
        [Tuple(HERO_DICT, Ma.Matrix)]
            (HERO_DICT) of hero portraits segmented from a hero roster,
            (Ma.Matrix) of positions images were detected in
    """

    original_image_modifiable = image.copy()
    original_image_unmodifiable = image.copy()

    hero_dict: HERO_DICT = {}
    base_args = (original_image_modifiable, size_allowance_boundary)
    multi_valid: list[Dict[str, DO.DimensionsObject]] = []

    # if maxHeroes:
    # multi_valid.append(get_hero_contours(*baseArgs, dilate=True))
    del blur_args["hsv_range"]
    hsv_range = [
        np.array([0, 0, 0]), np.array([179, 255, 192])]
    multi_valid.append(get_hero_contours(
        *base_args, hsv_range=hsv_range, **blur_args))
    # (hMin = 19 , sMin = 0, vMin = 36), (hMax = 179 , sMax = 255, vMax = 208)
    # hsv_range = [
    #     np.array([19, 0, 36]), np.array([179, 255, 208])]
    # multi_valid.append(get_hero_contours(
    #     *base_args, hsv_range=hsv_range, **blur_args))
    # (hMin = 0 , sMin = 0, vMin = 74), (hMax = 27 , sMax = 253, vMax = 255)
    base_args = (image.copy(), size_allowance_boundary)

    # (RMin = 67 , GMin = 55, BMin = 31), (RMax = 255 , GMax = 223, BMax = 169)
    blur_args["reverse"] = True

    # rgb_range = [np.array([67, 55, 31]), np.array([255, 223, 169])]
    # multi_valid.append(get_hero_contours(
    #     *base_args, rgb_range=rgb_range, **blur_args))

    hsv_range = [
        np.array([0, 0, 74]), np.array([27, 253, 255])]
    multi_valid.append(get_hero_contours(
        *base_args, hsv_range=hsv_range, **blur_args))

    hero_widths = []
    hero_heights = []

    for _heroes_list in multi_valid:
        for _object_name, _dimension_object in _heroes_list.items():
            hero_widths.append(_dimension_object.width)
            hero_heights.append(_dimension_object.height)
    hero_widths.sort()
    hero_heights.sort()

    hero_w_median = statistics.median(hero_widths)
    hero_h_median = statistics.median(hero_heights)

    spacing = round((hero_w_median + hero_h_median)/10)
    image_height, image_width = image.shape[:2]
    hero_matrix = MA.Matrix(image_height, image_width, spacing=spacing)
    for _hero_list in multi_valid:
        for _object_name, _dimension_object in _hero_list.items():

            hero_matrix.auto_append(_dimension_object, _object_name)
    # Sort before pruning so all columns get generated
    hero_matrix.sort()
    hero_matrix.prune(threshold=row_eliminate)

    for _row_index, hero_row in enumerate(hero_matrix):
        # print("row({}) length: {}".format(_row_index, len(hero_row)))
        for _object_index, row_item in enumerate(hero_row):

            coord_tuple = row_item.dimensions.coords()
            # y_coord = row_item.dimensions.y

            # x2_coord = row_item.dimensions.x2
            # y2_coord = row_item.dimensions.y2

            _hero_name = row_item.name

            segmented_image = original_image_unmodifiable[coord_tuple.y1:
                                                          coord_tuple.y2,
                                                          coord_tuple.x1:
                                                          coord_tuple.x2]
            # load.display_image(segmented_hero, display=True)

            if si_adjustment:
                width = row_item.dimensions.width
                height = row_item.dimensions.height
                x_adjust = round(width * si_adjustment)
                y_adjust = round(height * si_adjustment)

                _new_x = max(round(coord_tuple.x1 - x_adjust), 0)
                _new_y = max(round(coord_tuple.y1 - y_adjust), 0)
                segmented_image_copy = (
                    original_image_unmodifiable[_new_y:coord_tuple.y2,
                                                _new_x:coord_tuple.x2])
                modifiable_segmented_image = segmented_image_copy.copy()
                blurred = blur_image(modifiable_segmented_image, reverse=True, hsv_range=[
                    np.array([4, 69, 83]), np.array([23, 255, 355])])

                new_contours = cv2.findContours(
                    blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                new_contours = imutils.grab_contours(new_contours)
                # Get largest contour
                new_contours = sorted(new_contours, key=cv2.contourArea,
                                      reverse=True)[0]
                (_contour_x, _contour_y,
                 contour_w, contour_h) = cv2.boundingRect(
                    new_contours)
                # if GV.DEBUG:
                # new_contours = [new_contours]
                # cv2.fillPoly(modifiable_ROI, new_contours, [255, 0, 0])
                # load.display_image(modifiable_ROI, display=True)

                # (dimensions, name)
                _temp_row_item = RI.RowItem(
                    DO.SegmentRectangle(coord_tuple.x2-contour_w,
                                        coord_tuple.y2-contour_h,
                                        contour_w,
                                        contour_h))

                collision_item_id = hero_row.check_collision(
                    _temp_row_item, size_allowance_boundary=0.07,
                    avg_height=hero_matrix.get_avg_height(),
                    avg_width=hero_matrix.get_avg_width(),
                    avg_value_boundary=True)
                if collision_item_id != -1:
                    merged_row_item = hero_row.get(
                        collision_item_id, id_lookup=True)
                    _hero_name = merged_row_item.name

                w_border_offset = max(
                    round(0.03 * merged_row_item.dimensions.width), 2)
                h_border_offset = max(
                    round(0.03 * merged_row_item.dimensions.height), 2)
                merged_row_item.dimensions.x = max(
                    merged_row_item.dimensions.x - w_border_offset, 0)
                merged_row_item.dimensions.y = max(
                    merged_row_item.dimensions.y - h_border_offset, 0)

                # merged_row_item.dimensions._display(GV.IMAGE_SS,
                #                                      display=True)

                segmented_image = original_image_unmodifiable[
                    merged_row_item.dimensions.y:
                    merged_row_item.dimensions.y2,
                    merged_row_item.dimensions.x:
                    merged_row_item.dimensions.x2]
                # load.display_image([segmented_image, new_ROI], display=True,
                #                    multiple=True)
                # segmented_image = new_ROI
                if GV.DEBUG:
                    merged_vertex = merged_row_item.dimensions.coords()
                    cv2.rectangle(GV.IMAGE_SS,
                                  merged_vertex.vertex1(),
                                  merged_vertex.vertex2(),
                                  (255, 0, 0), 2)
            hero_dict[_hero_name] = {}
            if GV.verbosity(1):
                height, width = segmented_image.shape[:2]
                vertex_tuple = merged_row_item.dimensions.coords()
                cv2.rectangle(
                    GV.IMAGE_SS,
                    vertex_tuple.vertex1(),
                    vertex_tuple.vertex2(),
                    (0, 0, 0), 2)

            model_image_size = (GV.MODEL_IMAGE_SIZE, GV.MODEL_IMAGE_SIZE)
            segmented_image = cv2.resize(
                segmented_image,
                model_image_size,
                interpolation=cv2.INTER_CUBIC)
            hero_dict[_hero_name] = SegmentResult(
                _hero_name, segmented_image, row_item)

    return hero_dict, hero_matrix
