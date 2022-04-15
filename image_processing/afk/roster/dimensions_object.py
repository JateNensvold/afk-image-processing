"""
A module that wraps all features and code for the DimensionObject class

The DimensionsObject is used to wrap/interact with objects that existing in
space at a certain location, storing information such as the x and y
coordinates as well as the width and height of the object
"""
from typing import Any, NamedTuple, Union

import numpy as np

import image_processing.globals as GV
from image_processing.load_images import display_image
from image_processing.processing.types.types import SegmentRectangle
from pandas import BooleanDtype


class DoubleCoordinates(NamedTuple("DoubleCoordinates",
                                   [("x1", int),
                                    ("x2", int),
                                    ("y1", int),
                                    ("y2", int)])):
    """_summary_

    Args:
        Namedtuple (_type_): _description_
    """
    x1: int
    x2: int
    y1: int
    y2: int

    def __new__(cls: type["DoubleCoordinates"],
                x1: int, x2: int, y1: int, y2: int) -> "DoubleCoordinates":

        return super().__new__(cls, int(x1), int(x2), int(y1), int(y2))

    def vertex1(self):
        """_summary_

        Raises:
            IndexError: _description_

        Returns:
            _type_: _description_
        """

        return Coordinates(self.x1, self.y1)

    def vertex2(self):
        """_summary_

        Raises:
            IndexError: _description_

        Returns:
            _type_: _description_
        """

        return Coordinates(self.x2, self.y2)


class Coordinates(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """
    x: int
    y: int


class DimensionsObject:
    """
    Class used to store dimensions of on object such as x and y coordinates as
        well as width height and other useful pieces of information and helper
        methods
    """
    # pylint: disable=invalid-name

    def __str__(self):
        return f"DimensionsObject<x={self.x},y={self.y},w={self.width},h={self.height}>"

    @property
    def name(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # return f"<{self.x},y={self.y},w={self.width},h={self.height}>"
        return f"{self.x}x{self.y}_{self.width}x{self.height}"

    def __repr__(self) -> str:
        return str(self)

    def __init__(self, dimensions: SegmentRectangle, raw_data: Any = None,
                 full_number: bool = True):
        """
        Create a Dimensions object to hold x,y coordinates as well as
            width and height. Any changes made to x(x2) or y(y2)
            coordinates will update the corresponding 'width' or 'height' object
            for that dimension. Because of this capability always modify,
            x/x2/y/y2 when trying to change width and height.
        Args:
            dimensions (SegmentRectangle): named tuple
                containing (x,y, width, height)
            full_number: flag to enforce all 'DimensionObject' values to
                be whole numbers during its lifetime
        """
        self.__dict__["full_number"] = full_number
        self.full_number = full_number
        self.raw_data = raw_data
        self.x = dimensions.x
        self.y = dimensions.y

        self.x2 = self.x + dimensions.width
        self.y2 = self.y + dimensions.height

    def __getitem__(self, index: int) -> int:
        """
        Return dimension attribute found at 'index'

        Return:
            x on 0
            y on 1
            width on 2
            height on 3
        Raise:
            indexError on other
        """
        dimension_index = [self.x, self.y, self.width, self.height]
        try:
            return dimension_index[index]
        except IndexError as exception_handle:
            raise IndexError(
                f"DimensionObject index({index}) out of "
                f"range({len(dimension_index)})") from exception_handle

    def __setattr__(self, name: str, value: Union[int, float]):
        """
        Sets 'name' equal to 'value'
        If self.full_number is true all number types will be truncated to int
        """
        if isinstance(value, (int, float)) and self.full_number:
            # if self.full_number:  # pylint: disable=no-member
            self.__dict__[name] = int(value)
        else:
            self.__dict__[name] = value

    @property
    def width(self):
        """
        Calculate and return width of self
        """
        output = self.x2 - self.x
        if self.full_number:
            output = int(output)
        return output

    @property
    def height(self):
        """
        Calculate and return height of self
        """
        output = self.y2 - self.y
        if self.full_number:
            output = int(output)
        return output

    def merge(self, dimensions_object: "DimensionsObject",
              size_allowance_boundary=None, avg_value_boundary=False,
              avg_width=None, avg_height=None):
        """
        Combine two DimensionsObject into the existing DimensionsObject object
        Args:
            dimensions_object: DimensionsObject to absorb
            size_allowance_boundary: optional flag/int used to limit the size
                of a merge for any given coordinate. If the distance between
                the two merging coordinates are more than this number no merge
                will occur
            avg_value_boundary: flag to use size_allowance_boundary on
                'avg_width' and 'avg_height' arguments rather than the stored
                width and height
            avg_width: width value to calculate maximum size boundary change
                with, meant to simulate the "average" width of whatever this
                dimensional object is representing
            avg_height: height value to calculate maximum size boundary change
                with, meant to simulate the "average" height of whatever this
                dimensional object is representing
        """
        # width and height automatically update when setting x/x2/y/y2 values

        if size_allowance_boundary is not None:

            def print_fail(size_type, current_size, new_size, size_boundary,
                           size_change):
                if GV.verbosity(1):
                    print(
                        f"Failed to update {size_type} from ({current_size}) "
                        f"to ({new_size}) due to boundary "
                        f"limit({size_boundary}) < boundary "
                        f"size({size_change:.2f})")

            new_x = min(self.x, dimensions_object.x)
            new_y = min(self.y, dimensions_object.y)
            new_x2 = max(self.x2, dimensions_object.x2)
            new_y2 = max(self.y2, dimensions_object.y2)

            x_change = abs(new_x - self.x)
            y_change = abs(new_y - self.y)
            x2_change = abs(new_x2 - self.x2)
            y2_change = abs(new_y2 - self.y2)

            if avg_value_boundary:
                if avg_width:
                    max_width_change = size_allowance_boundary * avg_width
                if avg_height:
                    max_height_change = size_allowance_boundary * avg_height
            else:
                max_width_change = self.width * size_allowance_boundary
                max_height_change = self.height * size_allowance_boundary

            if x_change < max_width_change:
                self.x = new_x
            else:
                print_fail("x", self.x, new_x, size_allowance_boundary,
                           x_change/self.width)
            if y_change < max_height_change:
                self.y = new_y
            else:
                print_fail("y", self.y, new_y, size_allowance_boundary,
                           y_change/self.height)
            if x2_change < max_width_change:
                self.x2 = new_x2
            else:
                print_fail("x2", self.x2, new_x2, size_allowance_boundary,
                           x2_change/self.width)
            if y2_change < max_height_change:
                self.y2 = new_y2
            else:
                print_fail("y2", self.y2, new_y2, size_allowance_boundary,
                           y2_change/self.height)
        else:
            self.x = min(self.x, dimensions_object.x)
            self.y = min(self.y, dimensions_object.y)
            self.x2 = max(self.x2, dimensions_object.x2)
            self.y2 = max(self.y2, dimensions_object.y2)

    def coords(self):
        """
        Get coordinates of dimension object

        Return:
            [DoubleCoordinates]: named tuple containing (x1, x2, y1, y2)
        """
        return DoubleCoordinates(self.x, self.x2, self.y, self.y2)

    def within(self, dimension_object: Union["DimensionsObject", SegmentRectangle],
               size_difference: int, both: bool = False):
        """
        Check if 'dimension_object' and 'self' have a width and height that
            are no more than 'size_difference' apart

        Args:
            dimension_object (Union["DimensionsObject", SegmentRectangle]): dimension object to compare
                size against
            size_difference (int): percentage as a decimal
                ex. 0.3 would mean the two objects need to be within 30% of
                    each other
        Returns:
            bool: true when they are within 'size_difference' of each other
                false otherwise
        """
        width_adjustment = dimension_object.width * (size_difference)
        lower_width_bound = dimension_object.width - width_adjustment
        upper_width_bound = dimension_object.width + width_adjustment

        height_adjustment = dimension_object.height * (size_difference)
        lower_height_adjustment = dimension_object.height - height_adjustment
        upper_height_adjustment = dimension_object.height + height_adjustment

        lower_bound_valid = lower_width_bound <= self.width <= upper_width_bound
        upper_bound_valid = lower_height_adjustment <= self.height <= upper_height_adjustment
        if both:
            return lower_bound_valid and upper_bound_valid
        else:
            return lower_bound_valid or upper_bound_valid

    def dimensional_values(self):
        """
        Return the coordinates and width/height values for the dimension object

        Returns:
            [SegmentRectangle]: named tuple containing (x, y, width, height)
        """
        return SegmentRectangle(self.x, self.y, self.width, self.height)

    @property
    def size(self):
        """
        Return the size of the dimension object
        """
        return self.height * self.width

    def overlap(self, dim_object: "DimensionsObject"):
        """
        Calculate the area between 'dim_object' and the current object
        Args:
            dim_object: object to calculate overlap with
        Return:
            area(int) of overlap()
        """
        x_width = max(0, min(self.x2, dim_object.x2) -
                      max(self.x, dim_object.x))
        y_height = max(0, min(self.y2, dim_object.y2) -
                       max(self.y, dim_object.y))
        return x_width * y_height

    def _overlap_percent(self, dim_object: "DimensionsObject"):
        """
        Wrapper around 'overlap' method that returns raw overlap size as well
            as the percent 'dim_object' overlaps with itself
        Args:
            dim_object: object to calculate overlap with
        Return:
            tuple(int, float, float) of area of overlap, self percent overlap,
                and 'dim_object' percent overlap
        """
        raw_overlap = self.overlap(dim_object)
        return (raw_overlap, raw_overlap/self.size,
                raw_overlap/dim_object.size)

    def _display(self, source_image: np.ndarray, *args, **kwargs):
        display_object = source_image[self.y:
                                      self.y2,
                                      self.x:
                                      self.x2]
        display_image(display_object, *args, **kwargs)

    def create_image(self, source_image: np.ndarray):
        """_summary_

        Args:
            source_image (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        image_segment = source_image[self.y:
                                     self.y2,
                                     self.x:
                                     self.x2]
        return image_segment
