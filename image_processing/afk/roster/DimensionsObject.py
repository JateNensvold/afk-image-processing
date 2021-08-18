import image_processing.globals as GV
import image_processing.load_images as load
import numpy as np


class DimensionsObject:

    def __str__(self):
        return "({},{},{},{})".format(
            self.x, self.y, self.w, self.h)

    def __init__(self, dimensions, full_number: bool = True):
        """
        Create a Dimensions object to hold x,y coordinates as well as
            width and height. Any changes made to x(x2) or y(y2)
            coordinates will update the corresponding 'w' or 'h' object
            for that dimension. Because of this capability always modify,
            x/x2/y/y2 when trying to change width and height.
        Args:
            dimensions: tuple containing (x,y, width, height)
            full_number: flag to enforce all 'DimensionObject' values to
                be whole numbers during its lifetime
        """
        self.__dict__["full_number"] = full_number
        self.x = dimensions[0]
        self.y = dimensions[1]
        self.w = dimensions[2]
        self.h = dimensions[3]
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h

    def __getitem__(self, index: int) -> int:
        """
        Return dimension attribute found at 'index'

        Return:
            x on 0
            y on 1
            w on 2
            h on 3
        Raise:
            indexError on other
        """
        dimension_index = [self.x, self.y, self.w, self.h]
        try:
            return dimension_index[index]
        except IndexError:
            raise IndexError(
                "DimensionObject index({}) out of range({})".format(
                    index, len(dimension_index)))

    def __setattr__(self, name, value):
        """
        Sets 'name' equal to value
        If self.full_number is true all number types will be truncated to int
        When setting x or y values, the corresponding 'w' or 'h' value will
            also be automatically updated
        """
        if isinstance(value, (int, float)):
            if self.full_number:
                self.__dict__[name] = int(value)
            if name in ["x", "x2"] and "x" in self.__dict__ and "x2" in\
                    self.__dict__:
                self.__dict__["w"] = self.x2 - self.x
            elif name in ["y", "y2"] and "y" in self.__dict__ and "y2" in\
                    self.__dict__:
                self.__dict__["h"] = self.y2 - self.y

        else:
            self.__dict__[name] = value

    def _truncate_values(self):
        """
        Private method to enforce each 'DimensionObject' value is a
            whole number
        """
        self.x = int(self.x)
        self.y = int(self.y)
        self.x2 = int(self.x2)
        self.y2 = int(self.y2)
        self.w = int(self.w)
        self.h = int(self.h)

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
                if GV.VERBOSE_LEVEL >= 1:
                    print("Failed to update {} from ({}) to ({}) due to"
                          " boundary limit({}) < boundary size({:.2f})".format(
                              size_type, current_size, new_size, size_boundary,
                              size_change))

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
                max_width_change = self.w * size_allowance_boundary
                max_height_change = self.h * size_allowance_boundary

            if x_change < max_width_change:
                self.x = new_x
            else:
                print_fail("x", self.x, new_x, size_allowance_boundary,
                           x_change/self.w)
            if y_change < max_height_change:
                self.y = new_y
            else:
                print_fail("y", self.y, new_y, size_allowance_boundary,
                           y_change/self.h)
            if x2_change < max_width_change:
                self.x2 = new_x2
            else:
                print_fail("x2", self.x2, new_x2, size_allowance_boundary,
                           x2_change/self.w)
            if y2_change < max_height_change:
                self.y2 = new_y2
            else:
                print_fail("y2", self.y2, new_y2, size_allowance_boundary,
                           y2_change/self.h)
        else:
            self.x = min(self.x, dimensions_object.x)
            self.y = min(self.y, dimensions_object.y)
            self.x2 = max(self.x2, dimensions_object.x2)
            self.y2 = max(self.y2, dimensions_object.y2)

    def coords(self, single=True) -> list:
        """
        return top left and bottom right coordinate pairs

        Args:
            single: flag to return coordinates as single list
        Return:
            On single=True

            list(x1,y1,x2,y2)

            otherwise list of tuples

            list(TopLeft(x1,y1), BottomRight(x2,y2))
        """
        if single:
            return [self.x, self.y, self.x2, self.y2]
        else:
            return [(self.x, self.y), (self.x2, self.y2)]

    def dimensional_values(self, single=True):
        if single:
            return [self.x, self.y, self.w, self.h]
        else:
            return [(self.x, self.y), (self.w, self.h)]

    def size(self):
        """
        Return the size of the dimension object
        """
        return self.h * self.w

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
        return (raw_overlap, raw_overlap/self.size(),
                raw_overlap/dim_object.size())

    def _display(self, image: np.ndarray, *args, **kwargs):
        ROI = image[self.y:
                    self.y2,
                    self.x:
                    self.x2]
        load.display_image(ROI, *args, **kwargs)
