"""
A module that wraps all features and code for the ColumnObjects class

The ColumnObjects is used to wrap a list of DimensionObjects allowing for
interactions with groupings of DimensionObjects and track how each
DimensionObject is related to others in the grouping
"""
from typing import TYPE_CHECKING

import cv2
import rtree
import numpy as np

import image_processing.load_images as load
import image_processing.afk.roster.dimensions_object as DO

if TYPE_CHECKING:
    import image_processing.afk.roster.matrix as MA
    import image_processing.afk.roster.RowItem as RI


class ColumnObjects:
    """
    Class Wrapper around a list of dimensional Column Objects
        (i.e list of 1 dimensional objects) used to detect overlap between
        Column Objects
    """

    def __str__(self):

        return "".join([str((_c.x, _c.x2)) for _c in self.columns])

    def __init__(self, matrix: "MA.Matrix"):
        """
        Create a Columns object used to track the different columns in a
            'image_processing.stamina.matrix'
        Args:
            matrix: reference to partent matrix object
        """
        self.column_rtree = rtree.index.Index(interleaved=False)
        self.matrix = matrix
        self.columns: list[DO.DimensionsObject] = []
        self.column_id_to_index: dict[int, int] = {}
        self.column_id_to_column: dict[int, DO.DimensionsObject] = {}

    def __getitem__(self, index: int):
        """
        Get a row by its index
        Args:
            index: position of row in matrix

        Returns:
            image_processing.stamina.row
        """
        return self.columns[index]

    def find_column(self, row_item: "RI.RowItem", auto_add: bool = True,
                    update_rtree: bool = True,
                    conflict_resolve: bool = True,
                    new_column_thresh: int = 0.7):
        """
        Find the column that row_item would fall under

        Args:
            row_item: RowItem object to find column for
            auto_add: flag to automatically create and add a column to
                'ColumnObjects' when one is not found for row_item
            update_rtree: flag to automatically expand column sizes on
                intersections to the union of the column and 'row_item'
                dimensions
            conflict_resolve: flag that resolves conflicts when intersections
                return multiple columns, causes column with the most overlap
                with 'row_item' to be returned
            new_column_thresh: threshold to create a new column with if the
                overlap percent is under it
        Return:
            an (int) representing the index of the column if one is
                found/created

            otherwise (None)
        Raise:
            Exception when multiple columns are intersected with
        """

        # TODO Find out why invalid y values cause intersections/overlap to
        #   break
        segment_rectangle = DO.SegmentRectangle(
            row_item.dimensions.x,
            0,
            row_item.dimensions.width,
            self.matrix.source_height)
        _temp_dimensions_object = DO.DimensionsObject(segment_rectangle)
        _intersections_list = list(self.column_rtree.intersection(
            _temp_dimensions_object.coords()))

        if len(_intersections_list) == 1:
            _intersection_id = _intersections_list[0]
            _column = self.column_id_to_column[_intersection_id]
            if update_rtree:
                overlap_raw = _temp_dimensions_object.overlap(
                    _column)
                overlap_p = overlap_raw / _temp_dimensions_object.size()

                index = self.column_id_to_index[id(_column)]
                if overlap_p == 1:
                    pass
                elif overlap_p > new_column_thresh:
                    index = self.update_column(_column, row_item)
                elif overlap_p <= new_column_thresh:
                    # TODO Add a way to treat partial overlapping columns
                    # print("raw: {} percent: {}".format(
                    #       overlap_raw, overlap_p))
                    # print("Overlap: {} {} {}".format(
                    #     overlap_p, _temp_dimensions_object.coords(),
                    #     _column.coords()))
                    # print(auto_add)
                    if auto_add:
                        index = self.add_column(row_item)
            return index
        elif len(_intersections_list) == 0:
            if auto_add:
                self.add_column(row_item)
            else:
                return None
        else:
            intersection_indexes = [
                self.column_id_to_index[_intersection_id]
                for _intersection_id in _intersections_list]
            if conflict_resolve:
                overlap_list: tuple[int, int] = []
                for _index in intersection_indexes:
                    _column = self.columns[_index]
                    overlap_raw = _temp_dimensions_object.overlap(
                        _column)
                    overlap_p = overlap_raw / _temp_dimensions_object.size()
                    overlap_list.append((_index, overlap_p))

                # print(overlap_list)
                max_tuple = max(
                    overlap_list, key=lambda overlap_tuple: overlap_tuple[1])

                overlap_p = max_tuple[1]
                if overlap_p <= 1-new_column_thresh:
                    if auto_add:
                        index = self.add_column(row_item, resize=True)
                    else:
                        return None
                else:
                    index = max_tuple[0]
                # columns = [_c.coords() for _c in self.columns]
                # print(index, columns,
                #       len(self.columns))
                return index
            else:
                raise Exception(
                    "More than one intersection occurred "
                    f"between {row_item} and {intersection_indexes}")

    def balance(self):
        """
        Balance the size and number of all columns in self.columns
        If any gaps exists, create a new column in that area.
        Resize all columns to be the same width and ensure that consecutive
        columns are not overlapping
        """
        self._sort()
        min_x = self.columns[0].x
        max_x = self.columns[-1].x2

        columns_width = max_x - min_x
        avg_column_width = sum(
            [_i.width for _i in self.columns])/len(self.columns)
        num_columns = int(columns_width/avg_column_width)
        adjusted_avg_column_w = avg_column_width * 0.75
        last_min = self.columns[0].x2
        missing_columns = 0
        for _c in self.columns:
            if _c.x - last_min > adjusted_avg_column_w:
                missing_columns += 1
            last_min = _c.x2
        num_columns += missing_columns
        column_size = int(columns_width//num_columns)
        remainder = columns_width - column_size * \
            num_columns

        if num_columns != len(self.columns):
            for _itr in range(num_columns - len(self.columns)):
                self.columns.append(DO.DimensionsObject(
                    DO.SegmentRectangle(0, 0, 0, self.matrix.source_height)))
        for _column in self.columns:
            self.column_rtree.delete(id(_column), _column.coords())
            _column.x = min_x
            if remainder > 0:
                _column.x2 = min_x + column_size + 1
                remainder -= 1
            else:
                _column.x2 = min_x + column_size
            min_x = _column.x2
            self.column_rtree.insert(id(_column), _column.coords())
        # Add any changed/new items to column_id_to_index
        self._sort()

    def _display(self, image: np.ndarray, *args, **kwargs):
        # ROI = image[self.y:
        #             self.y2,
        #             self.x:
        #             self.x2]
        image_list = []
        for _d in self.columns:
            sized_image = image[_d.y:_d.y2, _d.x:_d.x2]
            height, width = sized_image.shape[:2]
            cv2.rectangle(sized_image, (0, 0), (width, height), (0, 0, 0), 2)
            image_list.append(sized_image)
        load.display_image(image_list, *args, multiple=True, ** kwargs)

    def _sort(self):
        """
        Sort columns in ascending order, and update 'column_id_to_index with
            new column indexes
        """
        self.columns.sort(key=lambda f: f.x)
        for _index, _column_object in enumerate(self.columns):
            self.column_id_to_index[id(_column_object)] = _index
            self.column_id_to_column[id(_column_object)] = _column_object

    def update_column(self, column: DO.DimensionsObject,
                      row_item: "RI.RowItem"):
        """
        Update 'column' with the x_coordinate union between 'row_item'
            and itself
        Args:
            column: column to update
            row_item: row_item to take x_coordinate union with
        """
        column_id = id(column)

        self.column_rtree.delete(column_id, column.coords())
        column.merge(row_item.dimensions)
        self.column_rtree.insert(column_id, column.coords())
        return self.column_id_to_index[column_id]

    def add_column(self, row_item: "RI.RowItem", resize: bool = False):
        """
        Add a new column to the ColumnObjects instance

        Args:
            row_item: RowItem object to build new column around
            resize: flag to resize existing columns if there is an
                intersection between 'row_item and another column and split
                the intersection difference each column
        Return:
            index(int) of new column added
        """
        column_dimensions_object = DO.DimensionsObject(
            DO.SegmentRectangle(row_item.dimensions.x,
                                0,
                                row_item.dimensions.width,
                                self.matrix.source_height))
        new_column_id = id(column_dimensions_object)
        if resize:
            _intersections_list = list(self.column_rtree.intersection(
                column_dimensions_object.coords()))
            _intersection_columns = [
                self.column_id_to_column[_intersection_id]
                for _intersection_id in _intersections_list]
            _intersection_columns.append(column_dimensions_object)
            min_x = min(_intersection_columns, key=lambda _f: _f.x).x
            max_x = max(_intersection_columns, key=lambda _f: _f.x2).x2
            columns_width = max_x - min_x
            column_size = int(columns_width//len(_intersection_columns))
            remainder = columns_width - column_size * \
                len(_intersection_columns)
            self.column_id_to_index[new_column_id] = len(
                self.column_id_to_index)
            self.column_id_to_column[new_column_id] = column_dimensions_object
            self.columns.append(column_dimensions_object)
            for _column in _intersection_columns:
                self.column_rtree.delete(id(_column), _column.coords())
                _column.x = min_x
                if remainder > 0:
                    _column.x2 = min_x + column_size + 1
                    remainder -= 1
                else:
                    _column.x2 = min_x + column_size
                min_x = _column.x2
                self.column_rtree.insert(id(_column), _column.coords())

        else:
            self.column_rtree.insert(
                new_column_id, column_dimensions_object.coords())
            self.column_id_to_index[new_column_id] = len(
                self.column_id_to_index)
            self.column_id_to_column[new_column_id] = column_dimensions_object

            self.columns.append(column_dimensions_object)
        self.columns.sort(key=lambda f: f.x)
        self._sort()
        return self.column_id_to_index[new_column_id]

    def _find_index(self, column_dimensions: DO.DimensionsObject) -> int:
        """
        *Only call to find index of dimensions in empty areas
        Find the forcasted index of an 'x_coord'

        Args:
            column_dimensions: DimensionObject to find
                forcasted column index of
        Return:
            forcasted index of column_dimensions
        """
        past_column_x = 0
        if len(self.columns) > 0:
            _index = 0
            while _index <= (len(self.columns) - 1):
                current_x_coord = self.columns[_index].x
                if column_dimensions.x > past_column_x and \
                        column_dimensions.x2 < current_x_coord:
                    return _index
                past_column_x = self.columns[_index].x2
                _index += 1
            return _index
        else:
            return 0
