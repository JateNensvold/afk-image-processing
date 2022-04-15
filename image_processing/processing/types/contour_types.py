
from typing import List, NamedTuple, Sequence, Tuple

import cv2
from numpy import int32

from image_processing.afk.roster.dimensions_object import DimensionsObject
from image_processing.processing.types.types import SegmentRectangle


COORDINATE = Tuple[int32, int32]
COORDINATE_WRAPPER = Tuple[COORDINATE]
CONTOUR = Sequence[COORDINATE_WRAPPER]
CONTOUR_LIST = Sequence[CONTOUR]

HIERARCHY = Tuple[int32, int32, int32, int32]
HIERARCHY_LIST = Sequence[HIERARCHY]
HIERARCHY_OUTER = Sequence[HIERARCHY_LIST]
HIERARCHY_RELATIONSHIP = Sequence[HIERARCHY_OUTER]


class ContourTuple(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """

    x_coordinate: int32
    y_coordinate: int32


class ContourCoordinate(ContourTuple):
    """_summary_

    Args:
        ContourTuple (_type_): _description_
    """
    x_coordinate: int32
    y_coordinate: int32

    def __init__(self, contour: COORDINATE_WRAPPER):
        """
        Dummy Method included for type hinting
        """

    def __new__(cls, contour: COORDINATE_WRAPPER):
        """_summary_

        Args:
            contour (CONTOUR): _description_
        """
        return super().__new__(cls, contour[0][0], contour[0][1])


class Contour:
    """_summary_
    """

    def __init__(self, image_contour: "ImageContours",
                 contour_index: int,
                 raw_contour: CONTOUR,
                 raw_hierarchy: HIERARCHY):
        """_summary_

        Args:
            image_contour (ImageContours): _description_
            raw_contour (CONTOUR): _description_
            raw_hierarchy (HIERARCHY): _description_
        """
        self._image_contour = image_contour
        self._contour: List[ContourCoordinate] = []
        self._contour_index = contour_index

        self.raw_contour = raw_contour
        self.dimension_object = DimensionsObject(
            SegmentRectangle(*cv2.boundingRect(raw_contour)))

        self._next_contour: int32 = raw_hierarchy[0]
        self._prev_contour: int32 = raw_hierarchy[1]
        self._child_contour: int32 = raw_hierarchy[2]
        self._parent_contour: int32 = raw_hierarchy[3]
        for contour_coordinate in raw_contour:
            self._contour.append(ContourCoordinate(contour_coordinate))

    def parent_of(self, child_index: int):
        """
        Check if the current Contour is the parent of another contour given
            the index of the child
        Args:
            child_index (int): contour index of child

        Returns:
            bool: True if child_index represents a contour that is a child
                of self, False otherwise
        """
        current_child = self.child_contour
        while current_child is not None:

            if current_child._contour_index == child_index:
                return True

            current_child = current_child.next_contour
        return False

    @property
    def next_contour(self):
        """_summary_
        """
        if self._next_contour == -1:
            return None
        return self._image_contour[self._next_contour]

    @property
    def prev_contour(self):
        """_summary_
        """
        if self._prev_contour == -1:
            return None
        return self._image_contour[self._prev_contour]

    @property
    def child_contour(self):
        """_summary_
        """
        if self._child_contour == -1:
            return None
        return self._image_contour[self._child_contour]

    @property
    def parent_contour(self):
        """_summary_
        """
        if self._parent_contour == -1:
            return None
        return self._image_contour[self._parent_contour]

    def __iter__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self

    def __next__(self) -> ContourCoordinate:
        """_summary_
        """
        return next(self._contour)

    def __getitem__(self, val: int):
        """_summary_

        Args:
            val (Union[int, Slice]): _description_
        """

        return self._contour[val]

    def __len__(self):

        return len(self._contour)


class ImageContours:
    """_summary_
    """

    def __init__(self, contour_list: CONTOUR_LIST,
                 hierarchy_list: HIERARCHY_RELATIONSHIP):
        """_summary_

        Args:
            contour_list (CONTOUR_LIST): _description_
        """

        self._contours: List[Contour] = []

        for contour_index, contour in enumerate(contour_list):
            self._contours.append(
                Contour(self, contour_index, contour, hierarchy_list[0][contour_index]))

    def __getitem__(self, val: int):
        """_summary_

        Args:
            val (Union[int, Slice]): _description_
        """

        output: Contour = self._contours[val]
        return output

    def __len__(self):

        return len(self._contours)
