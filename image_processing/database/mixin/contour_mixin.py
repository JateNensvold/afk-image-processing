from typing import Dict, List

import numpy as np
import cv2

from image_processing.processing.types.contour_types import Contour
from image_processing.processing.image_processing import HSVRange
from image_processing.afk.roster.dimensions_object import DimensionsObject
from image_processing.processing.types.types import SegmentRectangle
from image_processing.afk.hero.process_heroes import get_hero_contours
from image_processing.utils.pixel_kmeans import PixelKMeans


class ContourMixin:
    """_summary_
    """

    def create_contour_dict(self, image: np.ndarray, hsv_range: List[HSVRange]):
        """_summary_

        Args:
            image (np.ndarray): _description_
            hsv_range (List[HSVRange]): _description_

        Returns:
            _type_: _description_
        """
        hsv_result_dict: Dict[int, List[Contour]] = {}
        image_dimensions = DimensionsObject(
            SegmentRectangle(0, 0, *image.shape[:2]))
        for ascension_hsv_range in hsv_range:

            contour_wrapper = get_hero_contours(
                image, hsv_range=ascension_hsv_range)

            contour_list = contour_wrapper.largest(5)
            filtered_contour_list: List[Contour] = []
            bounding_box = DimensionsObject(SegmentRectangle(0, 0, 0, 0))
            for contour_instance in contour_list:
                if image_dimensions.within(contour_instance.dimension_object, 0.4):
                    bounding_box.merge(contour_instance.dimension_object)
                    filtered_contour_list.append(contour_instance)

            if len(filtered_contour_list) > 0:
                hsv_result_dict[bounding_box.size] = filtered_contour_list

        return hsv_result_dict

    def draw_contour(self, image: np.ndarray, contour_list: List[Contour]):
        """_summary_

        Args:
            image (np.ndarray): _description_
            contour_list (List[Contour]): _description_
        """
        contour_color = 255
        ascension_border_mask = np.zeros(image.shape[:2], np.uint8)

        if len(contour_list) > 0:
            largest_contour = contour_list[0]

            # Draw outermost/parent contour
            cv2.drawContours(
                ascension_border_mask,
                [largest_contour.raw_contour],
                -1, contour_color, -1)
            for contour_dimension_object in contour_list[1:]:
                # pylint: disable=protected-access
                if largest_contour._contour_index == \
                        contour_dimension_object._parent_contour:
                    contour_color = 0
                else:
                    contour_color = 255
                # Mask inner contours within "parent" contour, draw other
                #   contours normally
                cv2.drawContours(
                    ascension_border_mask,
                    [contour_dimension_object.raw_contour],
                    -1, contour_color, -1)

        merged_image = cv2.bitwise_and(
            image, image, mask=ascension_border_mask)

        centroids = self.find_dominate(merged_image)

        return centroids, ascension_border_mask

    def find_dominate(self, image: np.ndarray):
        """_summary_

        Args:
            image (np.ndarray): _description_
        """

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        kmeans = PixelKMeans()
        centroids = kmeans.fit_image(img)
        return centroids
