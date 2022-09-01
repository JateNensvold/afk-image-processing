"""
Module that contains most of the Screenshot/Roster parsing capabilities

Used to split apart the raw image passed from the CLI to processable
subsections that can be fed to the Models used to detect AFK Arena Hero
Attributes
"""
from typing import Any, Dict, List, Tuple

import cv2
from image_processing.processing.types.types import SegmentRectangle
from rtree.index import Index
import imutils
# Need this import to use imutils.contours.sort_contours,
#   without it Module raises AttributeError
# pylint: disable=unused-import
from imutils import contours  # noqa
from numpy import array, ndarray

import image_processing.globals as GV
from image_processing.afk.roster.dimensions_object import (DimensionsObject)
from image_processing.afk.roster.matrix import Matrix
from image_processing.afk.roster.RowItem import RowItem
from image_processing.processing.types.contour_types import (
    CONTOUR_LIST, HIERARCHY_RELATIONSHIP, Contour, ImageContours)
from image_processing.processing.image_data import SegmentResult
from image_processing.processing.image_processing_utils import HSVRange, blur_image
from image_processing.utils.utils import list_median

# pylint: disable=invalid-name
HERO_DICT = Dict[str, SegmentResult]


class LineSegment():
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, point1: int, point2: int):
        """_summary_

        Returns:
            _type_: _description_
        """
        self.point1 = point1
        self.point2 = point2

    def is_valid(self, new_point: int):
        """
        Check if 'new_point' lies on the line segment

        Args:
            new_point (int): location of point
            boundary_extension (int): percentage to expand point1 and point2
                outward by
        Returns:
            bool: true if point is on the line, false otherwise
        """

        return self.point1 <= new_point <= self.point2


class ContoursContainer:
    """_summary_
    """

    def __init__(self, image: ndarray):
        """_summary_

        Args:
            image (ndarray): blurred, filtered and thresholded image
        """
        self.image = image

        self.raw_contours: ImageContours = []
        self.bounding_box = DimensionsObject(SegmentRectangle(0, 0, 0, 0))
        self.contours = self._find_contours(image)

    def _find_contours(self, image: ndarray):
        """
        Find contours in image and wrap them in Dimension Object before
        returning them as a list

        Args:
            image (ndarray): image to find contours on
        Returns:
            (List[DimensionObject]): list of contours wrapped
                by DimensionObjects
        """

        self.raw_contours = ImageContours(*cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

        contour_list: List[Contour] = []

        for detected_contour in self.raw_contours:

            self.bounding_box.merge(detected_contour.dimension_object)
            contour_list.append(detected_contour)

        contour_list = sorted(
            contour_list, key=lambda x: cv2.contourArea(x.raw_contour), reverse=True)

        return contour_list

    def filter_contours(self, dimension_difference: int = 0.2, min_size: int = 2500,
                        dimension_median_difference: int = 0.15):
        """
        Return a list of contours from image that fit criteria passed to
            this function

        Args:
            dimension_difference (int, optional): percent difference that a
                contours height and width can differ by
                ex 0.2 = height length can be 20% smaller or larger than
                height/width average((height+width)/2)
            minimum_size (int, optional) = minimum area that a contour must take
                up to get added into the index and get added to the average
                height/width lists
            dimension_median_difference (int, optional): percentage amount that a
                'contours' dimension can differ from the median dimension and
                still be considered a valid contour (returns contours that have
                boundary outlines that are close to square)
                ex 0.2 = contour can have a height/width 20% higher or lower
                than the average
        """

        rtree_build_index = Index(interleaved=False)
        contour_list: List[Contour] = []
        lower_boundary = 1.0 - dimension_median_difference
        upper_boundary = 1.0 + dimension_median_difference

        for contour_index, contour_object in enumerate(self.contours):
            dim_object = contour_object.dimension_object
            contour_coordinates = dim_object.coords()
            contour_intersections = list(rtree_build_index.intersection(
                contour_coordinates))
            # no contour intersections so add contour to rtree index
            if len(contour_intersections) == 0:
                rtree_build_index.insert(contour_index, contour_coordinates)

            dimension_length_difference = abs(
                dim_object.height - dim_object.width)
            average_dimension_length = ((
                dim_object.height + dim_object.width)/2)
            allowed_dimension_difference = (
                average_dimension_length * dimension_difference)
            if (dimension_length_difference < allowed_dimension_difference and
                    (dim_object.size) > min_size):
                contour_list.append(contour_object)

        median_height = list_median(
            [contour_instance.dimension_object.height for contour_instance in contour_list])
        median_width = list_median(
            [contour_instance.dimension_object.width for contour_instance in contour_list])

        height_segment = LineSegment(median_height * lower_boundary,
                                     median_height * upper_boundary)
        weight_segment = LineSegment(median_width * lower_boundary,
                                     median_width * upper_boundary)

        valid_contour_list: List[Contour] = []
        for contour_object in contour_list:
            if (height_segment.is_valid(contour_object.dimension_object.height) or
                    weight_segment.is_valid(contour_object.dimension_object.width)):
                valid_contour_list.append(contour_object)

        return valid_contour_list

    def largest(self, count: int = 1):
        """
        Get the first 'count' number of contours

        Returns:
            list[DimensionsObject]: 'count' number of contours
        """
        return self.contours[:count]


def get_hero_contours(image: ndarray, **blur_kwargs: Dict[str, Any]):
    """
    Args:
        image: hero screenshot in BGR format
    Return:
        dict with imageSize as key and
            image_processing.stamina.DimensionalObject's as values
    """

    dilated_image = blur_image(image, **blur_kwargs)
    contours_wrapper = ContoursContainer(dilated_image)

    return contours_wrapper


def get_heroes(roster_image: ndarray,  blur_args: dict,
               dimension_median_difference: int = 0.15,
               si_adjustment: int = 0.2,
               row_eliminate: int = 5,
               ) -> Tuple[HERO_DICT, Matrix]:
    """
    Parse a screenshot or image of an AFK arena hero roster into sub
        components that represent all the heroes in the image

    Args:
        roster_image: image/screenshot of hero roster
        dimension_median_difference: percentage that each 'contour' boundary must be
            within the average contour size
        si_adjustment: percent of the image dimensions to take on the left and
            top side of the image to ensure si30/40 capture during hero
            contour re-evaluation(False/None for no adjustment)
        row_eliminate: minimum row size to allow (helps eliminate false
            positives i.e. helps remove contours of similar size shapes as the
            median hero shape) that are near the same size as the median
            hero detection)
        blur_args: keyword arguments for `processing.blur_image` method

    Return:
        [Tuple(HERO_DICT, Ma.Matrix)]
            (HERO_DICT) of hero portraits segmented from a hero roster,
            (Ma.Matrix) of positions images were detected in
    """

    original_image_unmodifiable = roster_image.copy()
    hero_dict: HERO_DICT = {}
    contour_container_list: List[List[Contour]] = []

    blur_args["hsv_range"] = HSVRange(0, 0, 0, 179, 255, 192)
    contour_container_list.append(get_hero_contours(
        roster_image, **blur_args).filter_contours(
            dimension_median_difference=dimension_median_difference))

    blur_args["reverse"] = True
    blur_args["hsv_range"] = HSVRange(0, 0, 74, 27, 253, 255)
    contour_container_list.append(get_hero_contours(
        roster_image, **blur_args).filter_contours(
            dimension_median_difference=dimension_median_difference))

    image_height, image_width = roster_image.shape[:2]
    hero_matrix = Matrix(image_height, image_width,
                         spacing_percent=GV.MATRIX_ROW_SPACING_PERCENT)
    for hero_contour_container in contour_container_list:
        for contour_object in hero_contour_container:
            hero_matrix.auto_append(
                contour_object.dimension_object.dimensional_values(),
                contour_object.dimension_object.name)

    # Sort before pruning so all columns get generated
    hero_matrix.sort()
    hero_matrix.prune(threshold=row_eliminate)

    for hero_row in hero_matrix:
        for row_item in hero_row:
            coord_tuple = row_item.dimensions.coords()
            _hero_name = row_item.name

            contour_image = roster_image[coord_tuple.y1:
                                         coord_tuple.y2,
                                         coord_tuple.x1:
                                         coord_tuple.x2]

            if si_adjustment:
                x_adjust = round(row_item.dimensions.width * si_adjustment)
                y_adjust = round(row_item.dimensions.height * si_adjustment)

                _new_x = max(round(coord_tuple.x1 - x_adjust), 0)
                _new_y = max(round(coord_tuple.y1 - y_adjust), 0)
                segmented_image_unmodifiable = (
                    original_image_unmodifiable[_new_y:coord_tuple.y2,
                                                _new_x:coord_tuple.x2])
                modifiable_segmented_image = segmented_image_unmodifiable.copy()
                blurred = blur_image(modifiable_segmented_image, reverse=True,
                                     hsv_range=GV.HERO_PORTRAIT_OUTLINE_HSV)

                new_contours = cv2.findContours(
                    blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                new_contours = imutils.grab_contours(new_contours)
                # Get largest contour
                new_contours = sorted(new_contours, key=cv2.contourArea,
                                      reverse=True)[0]
                (_contour_x, _contour_y,
                 contour_w, contour_h) = cv2.boundingRect(
                    new_contours)

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

                contour_image = original_image_unmodifiable[
                    merged_row_item.dimensions.y:
                    merged_row_item.dimensions.y2,
                    merged_row_item.dimensions.x:
                    merged_row_item.dimensions.x2]

                if GV.DEBUG:
                    merged_vertex = merged_row_item.dimensions.coords()
                    cv2.rectangle(GV.IMAGE_SS,
                                  merged_vertex.vertex1(),
                                  merged_vertex.vertex2(),
                                  (255, 0, 0), 2)
            hero_dict[_hero_name] = {}
            if GV.DEBUG:
                vertex_tuple = merged_row_item.dimensions.coords()
                cv2.rectangle(
                    GV.IMAGE_SS,
                    vertex_tuple.vertex1(),
                    vertex_tuple.vertex2(),
                    (0, 0, 0), 2)

            model_image_size = (GV.MODEL_IMAGE_SIZE, GV.MODEL_IMAGE_SIZE)
            resized_contour_image = cv2.resize(
                contour_image,
                model_image_size,
                interpolation=cv2.INTER_CUBIC)
            hero_dict[_hero_name] = SegmentResult(
                _hero_name, resized_contour_image, row_item,
                original_image_unmodifiable)

    return hero_dict, hero_matrix
