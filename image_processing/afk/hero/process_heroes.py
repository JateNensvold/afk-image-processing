"""
Module that contains most of the Screenshot/Roster parsing capabilities

Used to split apart the raw image passed from the CLI to processable
subsections that can be fed to the Models used to detect AFK Arena Hero
Attributes
"""
from statistics import median as stat_median
from typing import Dict, Tuple

import cv2
from rtree.index import Index
import imutils
# Need this import to use imutils.contours.sort_contours,
#   without it Module raises AttributeError
# pylint: disable=unused-import
from imutils import contours  # noqa
from numpy import array, ndarray, median

import image_processing.globals as GV
from image_processing.load_images import display_image
from image_processing.afk.roster.dimensions_object import (
    DimensionsObject, SegmentRectangle)
from image_processing.afk.roster.matrix import Matrix
from image_processing.afk.roster.RowItem import RowItem
from image_processing.processing.types import CONTOUR, CONTOUR_LIST
from image_processing.processing.image_data import SegmentResult
from image_processing.processing.image_processing import blur_image


# pylint: disable=invalid-name
HERO_DICT = Dict[str, SegmentResult]


def get_hero_contours(image: array, size_allowance_boundary: float,
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
        display_image(dilate)

    # Find contours
    contours_hierarchy_tuple = cv2.findContours(
        dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours: CONTOUR_LIST = imutils.grab_contours(
        contours_hierarchy_tuple)
    # detected_contours[0] - contour
    # detected_contours[0][0] - list of points
    # detected_contours[0][0] - point, ndarray of len 2
    detected_contours = sorted(detected_contours, key=cv2.contourArea,
                               reverse=True)

    # Iterate through contours and filter for detected_hero
    image_number = 0
    sizes: dict[int, list[DimensionsObject]] = {}
    heights = []
    widths = []

    idx = Index(interleaved=False)

    for _index, detected_contour in enumerate(detected_contours):

        image_segment_info = SegmentRectangle(
            *cv2.boundingRect(detected_contour))
        dim_object = DimensionsObject(image_segment_info, detected_contour)
        diff = abs(image_segment_info.height - image_segment_info.width)
        avg_h_w = ((image_segment_info.height + image_segment_info.width)/2)
        tolerance = avg_h_w * 0.2

        # if GV.DEBUG and display:
        idx_coords = dim_object.coords()
        intersections = list(idx.intersection(idx_coords))
        if not intersections:
            idx.insert(_index, idx_coords)
            if GV.DEBUG:
                split_coords = dim_object.coords()
                if split_coords.x1 == 198:
                    display_image(image, display=True)

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
    display_image(image, display=GV.DEBUG)

    h_mean = median(heights)
    w_mean = median(widths)

    lower_boundary = 1.0 - size_allowance_boundary
    upper_boundary = 1.0 + size_allowance_boundary
    h_low = h_mean * lower_boundary
    h_high = h_mean * upper_boundary
    w_low = w_mean * lower_boundary
    w_high = w_mean * upper_boundary

    occurrences = 0
    valid_sizes: Dict[str, DimensionsObject] = {}
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


def get_heroes(image: ndarray,  blur_args: dict,
               size_allowance_boundary: int = 0.15,
               si_adjustment: int = 0.2,
               row_eliminate: int = 5,
               ) -> Tuple[HERO_DICT, Matrix]:
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
    multi_valid: list[Dict[str, DimensionsObject]] = []

    # if maxHeroes:
    # multi_valid.append(get_hero_contours(*baseArgs, dilate=True))
    del blur_args["hsv_range"]
    hsv_range = [
        array([0, 0, 0]), array([179, 255, 192])]
    multi_valid.append(get_hero_contours(
        *base_args, hsv_range=hsv_range, **blur_args))
    # (hMin = 19 , sMin = 0, vMin = 36), (hMax = 179 , sMax = 255, vMax = 208)
    # hsv_range = [
    #     array([19, 0, 36]), array([179, 255, 208])]
    # multi_valid.append(get_hero_contours(
    #     *base_args, hsv_range=hsv_range, **blur_args))
    # (hMin = 0 , sMin = 0, vMin = 74), (hMax = 27 , sMax = 253, vMax = 255)
    base_args = (image.copy(), size_allowance_boundary)

    # (RMin = 67 , GMin = 55, BMin = 31), (RMax = 255 , GMax = 223, BMax = 169)
    blur_args["reverse"] = True

    # rgb_range = [array([67, 55, 31]), array([255, 223, 169])]
    # multi_valid.append(get_hero_contours(
    #     *base_args, rgb_range=rgb_range, **blur_args))

    hsv_range = [
        array([0, 0, 74]), array([27, 253, 255])]
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

    hero_w_median = stat_median(hero_widths)
    hero_h_median = stat_median(hero_heights)

    spacing = round((hero_w_median + hero_h_median)/10)
    image_height, image_width = image.shape[:2]
    hero_matrix = Matrix(image_height, image_width, spacing=spacing)
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
                    array([4, 69, 83]), array([23, 255, 355])])

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
                _temp_row_item = RowItem(
                    SegmentRectangle(coord_tuple.x2-contour_w,
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
