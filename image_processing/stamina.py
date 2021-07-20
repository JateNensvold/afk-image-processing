import os
import json
import cv2
import rtree

import numpy as np
import matplotlib.pyplot as plt
import image_processing.load_images as load
import image_processing.processing as processing
import image_processing.globals as GV
import imutils
# Need this import to use imutils.contours.sort_contours,
#   without it Module raises AttributeError
from imutils import contours  # noqa


class DimensionsObject:

    def __str__(self):
        return "({},{},{},{})".format(
            self.x, self.y, self.w, self.h)

    def __init__(self, dimensions, full_number: bool = True):
        """
            Create a Dimensions object to hold x,y coordinates as well as
                width and height
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
        if self.full_number:
            self._truncate_values()

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
        if isinstance(value, (int, float)):
            if self.full_number:
                self.__dict__[name] = int(value)
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

    def merge(self, dimensions_object: "DimensionsObject"):
        """
        Combine two DimensionsObject into the existing DimensionsObject object
        Args:
            dimensions_object: DimensionsObject to absorb
        """
        self.x = min(self.x, dimensions_object.x)
        self.y = min(self.y, dimensions_object.y)
        self.x2 = max(self.x2, dimensions_object.x2)
        self.y2 = max(self.y2, dimensions_object.y2)

        self.w = self.x2 - self.x
        self.h = self.y2 - self.y
        if self.full_number:
            self._truncate_values()

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


class ColumnObjects:

    def __init__(self, matrix: "matrix"):
        """
        Create a Columns object used to track the different columns in a
            'image_processing.stamina.matrix'
        Args:
            matrix: reference to partent matrix object
        """
        self.column_rtree = rtree.index.Index()
        self.matrix = matrix
        self.columns: list[DimensionsObject] = []
        self.column_id_to_index: dict[int, int] = {}
        self.column_id_to_column: dict[int, DimensionsObject] = {}

    def find_column(self, row_item: "RowItem", auto_add: bool = True,
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
        _temp_dimensions_object = DimensionsObject(
            (row_item.dimensions.x, 0,
             row_item.dimensions.w, self.matrix.source_height))
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
                    print("raw: {} percent: {}".format(
                          overlap_raw, overlap_p))
                    print("Overlap: {} {} {}".format(
                        overlap_p, _temp_dimensions_object.coords(),
                        _column.coords()))
                    print(auto_add)
                    if auto_add:
                        print("adding column intersection: {}".format(
                            overlap_p))
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
                    _intersection_object_dimensions = self.columns[_index]
                    overlap = row_item.dimensions.overlap(
                        _intersection_object_dimensions)
                    overlap_list.append((_index, overlap))
                # print(overlap_list)
                max_tuple = max(
                    overlap_list, key=lambda overlap_tuple: overlap_tuple[1])
                print(max_tuple)
                return max_tuple[0]
            else:
                raise Exception("More than one intersection occurred "
                                "between {} and {}".format(
                                    row_item, intersection_indexes))

    def update_column(self, column: DimensionsObject, row_item: "RowItem"):
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

    def add_column(self, row_item: "RowItem"):
        """
        Add a new column to the ColumnObjects instance

        Args:
            row_item: RowItem object to build new column around
        """
        columns = []
        for _i in self.columns:
            _temp_coords = _i.coords()
            columns.append(
                (_temp_coords[0], _temp_coords[2],  _temp_coords[1],
                 _temp_coords[3]))
        columns.sort(key=lambda f: f[0])

        column_dimensions_object = DimensionsObject(
            (row_item.dimensions.x, 0,
             row_item.dimensions.w, self.matrix.source_height))
        new_column_id = id(column_dimensions_object)
        self.column_rtree.insert(
            new_column_id, column_dimensions_object.coords())
        self.column_id_to_index[new_column_id] = len(self.column_id_to_index)
        self.column_id_to_column[new_column_id] = column_dimensions_object

        self.columns.append(column_dimensions_object)
        self.columns.sort(key=lambda f: f.x)
        for _index, _column_object in enumerate(self.columns):
            self.column_id_to_index[id(_column_object)] = _index
        return self.column_id_to_index[new_column_id]

    def _find_index(self, column_dimensions: DimensionsObject) -> int:
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


class RowItem():
    def __str__(self):
        return "({} {})".format(self.name, self.dimensions)

    def __init__(self, dimensions: tuple, name: str = None):
        """
        Create RowItem object from dimensions and name

        Args:
            dimensions: tuple containing (x,y, width, height)
            name: name of RowItem
        """
        self.dimensions = DimensionsObject(dimensions)
        if name is None:
            self.name = id(self)
        else:
            self.name = name
        self.alias = set()

        self.alias.add(id(self))
        self.column = None

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
        return self.dimensions[index]

    def merge(self, row_item: "RowItem"):
        """
        Combine two RowItem into the existing RowItem object
        Args:
            row_item: RowItem to absorb
        """
        self.dimensions.merge(row_item.dimensions)
        self.alias.add(row_item.name)

        self.alias.update(row_item.alias)
        self.alias.remove(id(row_item))


class row():

    def __init__(self, columns: "ColumnObjects"):
        """
        Create Row used to hold objects that have dimensions

        Args:
            columns: columnsObject used to calculate what column each rowItem
                is in
        """
        self._row_items_by_name: dict[str, RowItem] = {}
        self._row_items_by_id: dict[str, RowItem] = {}

        self._row_items: list[RowItem] = []
        self._idx = 0
        self.rtree = rtree.index.Index()
        self.columns = columns
        self.head: int = None
        self.avg_width = 0

    def __str__(self):
        return "({} ({}))".format(
            self.head,
            "".join([str(_row_item) for _row_item in self._row_items]))

    def __iter__(self):
        return self

    def __next__(self) -> RowItem:
        """
        Iterate over all RowItems in row
        """
        self._idx += 1
        try:
            return self._row_items[self._idx - 1]
        except IndexError:
            self._idx = 0
            raise StopIteration

    def __len__(self):
        return len(self._row_items)

    def get_head(self) -> int:
        """
        Gets top y coordinate of first item added to row, this y coordinate is
        used to anchor the top of the row to a fixed location

        Return:
            top y coordinate of original item added to row
        """
        return self.head

    def get(self, name: str, id_lookup=False) -> int:
        """
        Get an item by its associated name
        Args:
            name: name/id of RowItem to get
            id: flag to lookup RowItem by id instead of name
        Returns:
            RowItem associated with name
        """
        try:
            if id_lookup:
                return self._row_items_by_id[name]
            else:
                return self._row_items_by_name[name]
        except KeyError:
            if not id_lookup:
                for _row_item in self._row_items:
                    if name in _row_item.alias:
                        return self._row_items_by_name[_row_item.name]
            raise KeyError("Key '{}' not found in row '{}'".format(
                name, self.get_head()))

    def __getitem__(self, index: int) -> RowItem:
        """
        Get an item by its index
        Args:
            index: position of item in row
        Returns:
            image_processing.stamina.RowItem
        """
        return self._row_items[index]

    def _get_row_bottom(self):
        """
        Get the avg y coordinate of the bottom of the row.
        Return:
            avg y coordinate(float)
        """

        output = [_row_item.dimensions.y2 for _row_item in self._row_items]
        return np.mean(output)

    def _get_avg_height(self):
        """
        Get the avg height of items in the row
        Return:
            avg height (float)
        """
        output = [_row_item.dimensions.h for _row_item in self._row_items]
        return np.mean(output)

    def _get_item_gap(self):
        """
        Get avg gap between each item in row that is known to be consecutive
            i.e. there are no missing row_items between two existing row_items
        Return:
            Returns avg gap width between row_items when enough _row_items
                exist

            Otherwise returns None
        """
        item_gaps = []
        self.sort()
        for _row_item_index in range(len(self._row_items) - 1):
            _row_item1 = self._row_items[_row_item_index]
            _row_item2 = self._row_items[_row_item_index + 1]
            _temp_gap = (_row_item1.dimensions.x2 - _row_item2.dimensions.x)
            if _temp_gap < self.avg_width:
                item_gaps.append(_temp_gap)
        if len(item_gaps) > 0:
            avg_gap: int = np.mean(item_gaps)
            return avg_gap
        else:
            return None

    def _add_average(self, new_num: int) -> int:
        """
        Calculate running average width of row with 'new_num' updating the
            avg_width.
        Arg:
            new_num: number to update average with
        Return:
            Updated avg_width with 'new_num' added
        """
        output = self.avg_width + ((
            new_num - self.avg_width) / (len(self._row_items) + 1))
        return output

    def _remove_average(self, remove_num: int) -> int:
        """
        Calculate running average width of row with 'remove_num' removed from
            the avg_width.
        Arg:
            new_num: number to update average with
        Return:
            Updated avg_width with 'remove_num' removed
        """
        if len(self._row_items) == 1:
            return 0
        output = self.avg_width + ((
            self.avg_width - remove_num) / (len(self._row_items) - 1))
        return output

    def append(self, dimensions, name: str = None, detect_collision=True):
        """
        Adds a new RowItem to Row
        Args:
            dimensions: x,y,w,h of object
            name: identifier for row item, can be used for lookup later
            detect_collision: check for object overlap/collisions when
                appending to row
        Return:
            Index(int) RowItem was added at when successfully appended

            otherwise returns -1(int)
        """
        _temp_RowItem = RowItem(dimensions, name)
        if len(self._row_items) == 0:
            self.head = _temp_RowItem.dimensions.y
        if detect_collision:
            _collision_status = self.check_collision(_temp_RowItem)
            # If collision is successful, don't return
            if _collision_status != -1:
                return _collision_status
        _temp_id = id(_temp_RowItem)
        self._row_items_by_name[name] = _temp_RowItem
        self._row_items_by_id[_temp_id] = _temp_RowItem

        avg_width = self._add_average(_temp_RowItem.dimensions.w)
        self.avg_width = avg_width
        self._row_items.append(_temp_RowItem)

        self.rtree.insert(id(_temp_RowItem),
                          _temp_RowItem.dimensions.coords())
        self.columns.find_column(_temp_RowItem)
        return _collision_status

    def check_collision(self, new_row_item: RowItem,
                        size_allowance_boundary: int = 0.25,
                        resolve_error: bool = True):
        """
        Check if row_item's dimensions overlap with any of the objects
            in the row object, and merge collision_object with overlaping
            object if collisions is detected
        Args:
            row_item: new RowItem to check against existing RowItems
            size_allowance_boundary: percent size that collision image must be
                within the average of all other images in row
            resolve_error: flag to resolve errors when multi RowItem collisions
                occur and return the item with the greatest overlap
        Return:
            When collision occurs id(int) of updated row object is returned

            otherwise -1(int) is returned
        """

        _new_item_coords = new_row_item.dimensions.coords()
        _intersections_list = list(self.rtree.intersection(_new_item_coords))
        if len(_intersections_list) == 1:
            _intersection_id = _intersections_list[0]

            _collision_row_item = self._row_items_by_id[_intersection_id]
            _collision_item_coordinates = _collision_row_item.\
                dimensions.coords()

            old_coords = _collision_row_item.dimensions.coords()
            old_width = _collision_row_item.dimensions.w
            _collision_row_item.merge(new_row_item)

            new_coords = _collision_row_item.dimensions.coords()
            new_width = _collision_row_item.dimensions.w
            if new_coords != old_coords:
                self.rtree.delete(_intersection_id,
                                  _collision_item_coordinates)
                if old_width != new_width:
                    removed_width = self._remove_average(old_width)
                    self.avg_width = removed_width
                    removed_width = self._add_average(new_width)
                    self.avg_width = removed_width

                self.rtree.insert(_intersection_id,
                                  _collision_row_item.dimensions.coords())
            return _intersection_id
        elif len(_intersections_list) > 1:
            _intersection_objects = [
                self._row_items_by_id[_intersection_id]
                for _intersection_id in _intersections_list]
            if resolve_error:
                overlap_list: tuple[int, int] = []
                for _row_item in _intersection_objects:
                    overlap = _row_item.dimensions.overlap(
                        new_row_item.dimensions)
                    overlap_list.append((id(_row_item), overlap))
                # print(overlap_list)
                max_overlap_tuple = max(
                    overlap_list, key=lambda overlap_tuple: overlap_tuple[1])
                return max_overlap_tuple[0]

                # raise Exception("More than one intersection occurred "
                #                 "between {} and {}".format(
                #                     new_row_item,
                #                     _intersection_objects))
            raise Exception("More than one intersection occurred "
                  "between {} and {}".format(
                      new_row_item,
                      [str(_i) for _i in _intersection_objects]))
        else:
            # No intersection at all
            return -1

    def sort(self):
        '''
        Sort row by x coordinate of each RowItem
        '''
        self._row_items.sort(key=lambda _row_item: _row_item.dimensions.x)


class matrix():

    def __init__(self, source_height: int, source_width: int,
                 spacing: int = 10):
        """
        Create matrix object to track list of image_processing.stamina.row

        Args:
            source_height: maximum height of source
            source_width: maximum width of source
            spacing: minimum distance between each row without merging
        """
        self.source_height = source_height
        self.source_width = source_width
        self.spacing = spacing
        self._heads: dict[int, callable[[], int]] = {}
        self._row_list: list[row] = []
        self._idx = 0
        self.columns = ColumnObjects(self)

    def __str__(self):
        return "\n".join([_row for _row in self._row_list])

    def __iter__(self):
        """
        Return self
        """
        return self

    def __next__(self):
        """
        Iterate over all rows in self._row_list
        """
        self._idx += 1
        try:
            return self._row_list[self._idx - 1]
        except IndexError:
            self._idx = 0
            raise StopIteration

    def __len__(self):
        return len(self._row_list)

    def __getitem__(self, index: int):
        """
        Get a row by its index
        Args:
            index: position of row in matrix

        Returns:
            image_processing.stamina.row
        """
        return self._row_list[index]

    def get_avg_width(self) -> int:
        """
        Return the average width of RowItems in the matrix
        """
        avg_width = [_row.avg_width for _row in self._row_list]
        return np.mean(avg_width)

    def get_avg_height(self):
        """
        Return the average height of RowItems in the matrix
        """
        avg_height = [_row._get_avg_height() for _row in self._row_list]
        return np.mean(avg_height)

    def get_avg_row_gap(self):
        """
        Get the average width gap between RowItems in each row

        Return:
            Average width gap(int) between RowItems when matrix contains rows

            Otherwise returns (None)
        """
        gap_list: list[int] = []
        for _row in self._row_list:
            _temp_gap = _row._get_item_gap()
            if _temp_gap is not None:
                gap_list.append(_temp_gap)
        avg_gap = np.mean(gap_list)
        if avg_gap == 0:
            return None
        else:
            return avg_gap

    def auto_append(self, dimensions: tuple, name: str,
                    detect_collision: bool = True,
                    dimension_object=False):
        """
        Add a new entry into the matrix, either creating a new row or adding to
            an existing row depending on `spacing` settings and distance.
        Args:
            dimensions: x,y,w,h of object
            name: identifier for object
            detect_collision: check for object overlap/collisions when
                appending to row
            dimension_object: flag to treat `dimensions` as DimensionObject
                instead of a tuple
        Return:
            None
        """
        if dimension_object:
            _temp_dimensions = dimensions
        else:
            _temp_dimensions = DimensionsObject(dimensions)
        y = _temp_dimensions.y
        _row_index = None
        for _index, _head in self._heads.items():
            # If there are no close rows set flag to create new row
            if abs(_head() - y) < self.spacing:
                _row_index = _index
                break
        if _row_index is not None:
            self._row_list[_row_index].append(
                dimensions, name, detect_collision=detect_collision)
        else:
            _temp_row = row(self.columns)
            _temp_row.append(dimensions, name,
                             detect_collision=detect_collision)
            self._heads[len(self._row_list)] = _temp_row.get_head
            self._row_list.append(_temp_row)

    def sort(self):
        """
        Sort Matrix by y coordinate of each row
        """
        for _row in self._row_list:
            _row.sort()
        self._row_list.sort(key=lambda _row: _row.head)

    def prune(self, threshold: int, fill_hero: bool = True,
              hard_threshold: int = 1):
        """
        Remove all rows that have a length less than the `threshold`

        Args:
            threshold: limit that determines if a row should be pruned when
                its length is less than this
            fill_hero: flag to attempt to fill missing hero locations in each
                row that is below the threshold
            hard_threshold: number that will trigger a row removal regardless
                of if fill_hero is set to true
        Return:
            None
        """
        _prune_list = []
        for _row in self._row_list:
            for _row_item in _row:
                _row.columns.find_column(_row_item)

        for _index, _row_object in enumerate(self._row_list):
            if len(_row_object) < threshold and _index != (
                    len(self._row_list) - 1):
                if fill_hero and len(_row_object) > hard_threshold:
                    # print("before: {}".format(_row_object))
                    _column_index = 0
                    while _column_index < len(self.columns.columns):
                        columns = [self.columns.find_column(
                            _row_item) for _row_item in _row_object]
                        # print(_column_index, columns,
                        #       len(self.columns.columns))
                        if _column_index not in columns:
                            # print("{} not in {}".format(
                            #     _column_index, columns))
                            right_side = [
                                _i for _i in columns if _i > _column_index]
                            left_side = [
                                _i for _i in columns if _i < _column_index]
                            right_side.sort()
                            left_side.sort()
                            if len(left_side) > 0:
                                closest_left = left_side[-1]
                            else:
                                closest_left = None
                            if len(right_side) > 0:
                                closest_right = right_side[0]
                            else:
                                closest_right = None
                            if closest_left and closest_right:
                                left_row_object = self.columns.columns[
                                    closest_left]
                                right_row_object = self.columns.columns[
                                    closest_right]
                                left_x = left_row_object.x2
                                right_x = right_row_object.x
                                # print("x1: {} x2: {}".format(
                                # left_x, right_x))
                                gap_size = right_x - left_x
                                _avg_width = self.get_avg_width()
                                # print(gap_size, _avg_width)
                                missing_row_items = int(gap_size//_avg_width)
                                # print("num missing:{} gap:{} item_w:{}"
                                # .format(
                                #     missing_row_items, gap_size, _avg_width))

                                extra_gap = gap_size - \
                                    (missing_row_items*_avg_width)
                                extra_gap_number = (missing_row_items + 1)
                                extra_gap_width = extra_gap/extra_gap_number

                                _avg_height = self.get_avg_height()
                                for _itr in range(missing_row_items):
                                    _temp_dims = (
                                        left_x + extra_gap_width,
                                        _row_object._get_row_bottom(
                                        ) - _avg_height,
                                        _avg_width, _avg_height)
                                    _temp_dim_object = DimensionsObject(
                                        _temp_dims)
                                    left_x = _temp_dim_object.x2
                                    _row_object.append(_temp_dim_object)
                            elif closest_left:
                                _avg_height = self.get_avg_height()
                                _avg_width = self.get_avg_width()

                                left_row_object = self.columns.columns[
                                    closest_left]

                                _avg_gap = self.get_avg_row_gap()
                                left_x = left_row_object.x2
                                gap_size = self.source_width - left_x
                                _leftover_gap = gap_size

                                while _leftover_gap > _avg_width:
                                    _temp_dims = (
                                        left_x + _avg_gap,
                                        _row_object._get_row_bottom(
                                        ) - _avg_height,
                                        _avg_width, _avg_height)
                                    _temp_dim_object = DimensionsObject(
                                        _temp_dims)
                                    left_x = _temp_dim_object.x2
                                    _row_object.append(_temp_dim_object)
                                    _leftover_gap -= (_avg_width + _avg_gap)

                            elif closest_right:
                                # print(closest_right)
                                right_row_object = self.columns.columns[
                                    closest_right]

                                right_x = right_row_object.x
                                gap_size = right_x

                                _avg_height = self.get_avg_height()
                                _avg_width = self.get_avg_width()
                                _avg_gap = self.get_avg_row_gap()

                                _leftover_gap = gap_size
                                while _leftover_gap > _avg_width:
                                    _temp_dims = (
                                        right_x - _avg_width,
                                        _row_object._get_row_bottom(
                                        ) - _avg_height,
                                        _avg_width, _avg_height)

                                    _temp_dim_object = DimensionsObject(
                                        _temp_dims)
                                    right_x = _temp_dim_object.x
                                    _row_object.append(_temp_dim_object)
                                    _leftover_gap -= (_avg_width + _avg_gap)

                        _column_index += 1
                    # print("after: {}".format(_row_object))
                else:
                    _prune_list.append(_index)
            _row_object.sort()
        if len(_prune_list) > 0:
            if GV.VERBOSE_LEVEL >= 1:
                print("Deleting ({}) row objects({}) from matrix. Ensure that "
                      "getHeroes was successful".format(
                          len(_prune_list), _prune_list))
            for _index in sorted(_prune_list, reverse=True):
                if GV.VERBOSE_LEVEL >= 1:
                    print("Deleted row object ({}) of len ({})".format(
                        self._row_list[_index], len(self._row_list[_index])))
                self._row_list.pop(_index)
                del self._heads[_index]

    def detect_missing(self, index):
        pass


def cachedproperty(func):
    """
        Used on methods to convert them to methods that replace themselves
        with their return value once they are called.
    """
    if func.__name__ in GV.CACHED:
        return GV.CACHED[func.__name__]

    def cache(*args):
        result = func(*args)
        GV.CACHED[func.__name__] = result
        return result
    return cache


def get_stamina_area(rows: list, heroes: dict, sourceImage: np.array):
    numRows = len(rows)
    staminaCount = {}
    # iterate across length of row
    averageHeight = 0
    samples = 0
    for j in range(numRows):
        for i in range(len(rows[j])):
            # iterate over column
            # Last row
            unitName = rows[j][i][1]
            y = heroes[unitName]["dimensions"]["y"]
            x = heroes[unitName]["dimensions"]["x"]
            gapStartX = x[0]
            gapStartY = y[1]
            gapWidth = x[1] - x[0]
            if (j + 1) == numRows:
                gapBottom = gapStartY + int(averageHeight)
            else:
                gapBottom = heroes[rows[j+1][i][1]]["dimensions"]["y"][0]

                samples += 1
                a = 1/samples
                b = 1 - a
                averageHeight = (a * (gapBottom - gapStartY)
                                 ) + (b * averageHeight)
            staminaArea = sourceImage[gapStartY:gapBottom,
                                      gapStartX:gapStartX + gapWidth]
            staminaCount[unitName] = staminaArea

    return staminaCount


def get_text(staminaAreas: dict, train: bool = False):

    # build template dictionary
    digits = {}
    numbersFolder = GV.staminaTemplatesPath
    # numbersFolder = os.path.join(os.path.dirname(
    #     os.path.abspath(__file__)), "numbers")

    referenceFolders = os.listdir(numbersFolder)
    for folder in referenceFolders:
        if folder not in digits:
            digits[folder] = {}
        digitFolder = os.path.join(numbersFolder, folder)
        for i in os.listdir(digitFolder):
            name, ext = os.path.splitext(i)
            digits[folder][name] = cv2.imread(os.path.join(digitFolder, i))
    output = {}
    for name, stamina_image in staminaAreas.items():
        original = stamina_image.copy()

        lower = np.array([0, 0, 176])
        upper = np.array([174, 34, 255])
        hsv = cv2.cvtColor(stamina_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask == 0] = (255, 255, 255)

        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.threshold(
            result, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

        digit_contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_contours = imutils.grab_contours(digit_contours)
        digit_contours = imutils.contours.sort_contours(
            digit_contours,
            method="left-to-right")[0]
        digitText = []
        for digit in digit_contours:
            x, y, w, h = cv2.boundingRect(digit)
            if w > 6 and h > 12:
                ROI = stamina_image[y:y+h, x:x+w]
                sizedROI = cv2.resize(ROI, (57, 88))
                if train:
                    digitFeatures(sizedROI)
                else:
                    numberScore = []
                    for digitName, digitDICT in digits.items():
                        scores = []
                        for digitIteration, digitImage in digitDICT.items():
                            templateMatch = cv2.matchTemplate(
                                sizedROI, digitImage, cv2.TM_CCOEFF)
                            (_, score, _, _) = cv2.minMaxLoc(templateMatch)
                            scores.append(score)
                        avgScore = sum(scores)/len(scores)
                        numberScore.append((digitName, avgScore))
                    temp = sorted(
                        numberScore, key=lambda x: x[1], reverse=True)
                    digitText.append(temp[0][0])

        text = "".join(digitText)
        output[name] = text
    return output


@ cachedproperty
def signature_template_mask(templates: dict):
    siFolders = os.listdir(GV.siBasePath)
    si_dict = {}

    for folder in siFolders:
        SIDir = os.path.join(GV.siBasePath, folder)
        SIPhotos = os.listdir(SIDir)
        if folder == "40":
            continue
        for imageName in SIPhotos:

            siImage = templates[folder]["image"]
            # siImage = cv2.imread(os.path.join(
            #     GV.siBasePath, folder, imageName))
            # SIGray = cv2.cvtColor(siImage, cv2.COLOR_BGR2GRAY)

            templateImage = templates[folder].get(
                "crop", templates[folder].get("image"))
            mask = np.zeros_like(templateImage)

            if "morph" in templates[folder] and templates[folder]["morph"]:

                hsv_range = [np.array([0, 0, 206]), np.array([159, 29, 255])]

                thresh = processing.blur_image(
                    templateImage, hsv_range=hsv_range, reverse=True)

            else:
                templateGray = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)

                thresh = cv2.threshold(
                    templateGray, 0, 255,
                    cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

            inverted = cv2.bitwise_not(thresh)
            x, y = inverted.shape[:2]
            cv2.rectangle(inverted, (0, 0), (y, x), (255, 0, 0), 1)

            if folder == "0" or folder == "10":
                pass
            else:
                inverted = cv2.bitwise_not(inverted)

            siCont = cv2.findContours(
                inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            siCont = imutils.grab_contours(siCont)
            if folder == "0":

                siCont = sorted(siCont, key=cv2.contourArea, reverse=True)[
                    1:templates[folder]["contourNum"]+1]
            else:
                siCont = sorted(siCont, key=cv2.contourArea, reverse=True)[
                    :templates[folder]["contourNum"]]
            cv2.fillPoly(mask, siCont, [255, 255, 255])

            if folder not in si_dict:
                si_dict[folder] = {}
            # si_dict[folder]["image"] = SIGray
            si_dict[folder]["template"] = templateImage
            si_dict[folder]["source"] = siImage
            si_dict[folder]["mask"] = mask

    return si_dict


def signatureItemFeatures(hero: np.array,
                          si_dict: dict,
                          lvlRatioDict: dict = None):
    """
    Runs template matching SI identification against the 'hero' passed in.
        When lvlRatioDict is passed in the templates will be rescaled to
        attempt and find the best template size for detecting SI objects

    Args:
        hero: np.array(x,y.3) representing an rgb image
        templates: dictionary of information about each SI template to get ran
            against the image
        lvlRatioDict: dictionary that contains the predicted height of each
            signature item based on precomputed text to si scaling calculations
    Returns:
        dictionary with best "score" that each template achieved on the 'hero'
            image
    """
    x, y, _ = hero.shape
    x_div = 2.4
    y_div = 2.0
    hero_copy = hero.copy()

    crop_hero = hero[0: int(y/y_div), 0: int(x/x_div)]
    numberScore = {}

    for pixel_offset in range(-5, 50, 2):
        for folder_name, imageDict in si_dict.items():
            si_image = imageDict["template"]

            sourceSIImage = imageDict["source"]
            hero_h, hero_w = sourceSIImage.shape[:2]

            si_height, original_width = si_image.shape[:2]

            base_height_ratio = si_height/hero_h
            # resize_height
            base_new_height = round(
                lvlRatioDict[folder_name]["height"]) + pixel_offset
            new_height = round(base_new_height * base_height_ratio)
            scale_ratio = new_height/si_height
            new_width = round(original_width * scale_ratio)
            si_image = cv2.resize(
                si_image, (new_width, new_height))
            si_image_gray = cv2.cvtColor(si_image, cv2.COLOR_BGR2GRAY)
            hero_gray = cv2.cvtColor(crop_hero, cv2.COLOR_BGR2GRAY)

            mask = cv2.resize(
                imageDict["mask"], (new_width, new_height))
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask_gray = mask

            height, width = si_image.shape[:2]

            # sizedROI = cv2.resize(
            #     hero, (int(x * image_ratio), int(y * image_ratio)))
            if folder_name != "0":
                mask_gray = cv2.bitwise_not(mask_gray)
            # if folder_name == "10":
            # load.display_image(mask_gray, display=True)

            try:
                templateMatch = cv2.matchTemplate(
                    hero_gray, si_image_gray, cv2.TM_CCOEFF_NORMED,
                    mask=mask_gray)
            except Exception:
                if crop_hero.shape[0] < si_image.shape[0] or \
                        crop_hero.shape[1] < si_image.shape[1]:
                    _height, _width = si_image.shape[:2]
                    crop_hero = hero[0: max(int(y/y_div), int(_height*1.2)),
                                     0: max(int(x/x_div), int(_width*1.2))]

                    hero_gray = cv2.cvtColor(crop_hero, cv2.COLOR_BGR2GRAY)
                templateMatch = cv2.matchTemplate(
                    hero_gray, si_image_gray, cv2.TM_CCOEFF_NORMED,
                    mask=mask_gray)
            (_, score, _, scoreLoc) = cv2.minMaxLoc(templateMatch)
            coords = (scoreLoc[0] + width, scoreLoc[1] + height)

            if folder_name not in numberScore:
                numberScore[folder_name] = []
            numberScore[folder_name].append(
                (score, pixel_offset, (scoreLoc, coords)))
    best_score = {}
    for _folder, _si_scores in numberScore.items():
        numberScore[_folder] = sorted(_si_scores, key=lambda x: x[0])
        _best_match = numberScore[_folder][-1]
        _score_loc = _best_match[2][0]
        _coords = _best_match[2][1]

        cv2.rectangle(hero_copy, _score_loc, _coords, (255, 0, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(
            hero_copy, _folder, _coords, font, fontScale, color, thickness,
            cv2.LINE_AA)
        best_score[_folder] = round(_best_match[0], 3)
    # print(best_score)
    # load.display_image(hero_copy, display=True)
    return best_score


@ cachedproperty
def furniture_template_mask(templates: dict):

    fi_dict = {}
    fi_folders = os.listdir(GV.fi_base_path)

    for folder in fi_folders:
        fi_dir = os.path.join(GV.fi_base_path, folder)
        fi_photos = os.listdir(fi_dir)
        for image_name in fi_photos:
            fi_image = templates[folder]["image"]
            template_image = templates[folder].get(
                "crop", templates[folder]["image"])
            mask = np.zeros_like(template_image)

            if "morph" in templates[folder] and templates[folder]["morph"]:
                se = np.ones((2, 2), dtype='uint8')
                # inverted = cv2.bitwise_not(inverted)

                lower = np.array([0, 8, 0])
                upper = np.array([179, 255, 255])
                hsv = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)
                thresh = cv2.inRange(hsv, lower, upper)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)
                thresh = cv2.bitwise_not(thresh)

            else:
                template_gray = cv2.cvtColor(
                    template_image, cv2.COLOR_BGR2GRAY)

                thresh = cv2.threshold(
                    template_gray, 147, 255,
                    cv2.THRESH_BINARY)[1]
                inverted = thresh
            x, y = inverted.shape[:2]
            cv2.rectangle(inverted, (0, 0), (y, x), (255, 0, 0), 1)
            inverted = cv2.bitwise_not(inverted)

            fi_contours = cv2.findContours(
                inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            fi_contours = imutils.grab_contours(fi_contours)
            fi_contours = sorted(fi_contours, key=cv2.contourArea,
                                 reverse=True)[:templates[
                                     folder]["contourNum"]]
            cv2.drawContours(mask, fi_contours, -1,
                             (255, 255, 255), thickness=cv2.FILLED)

            if folder not in fi_dict:
                fi_dict[folder] = {}
            # si_dict[folder][imageName] = siImage
            fi_dict[folder]["template"] = template_image
            fi_dict[folder]["source"] = fi_image

            fi_dict[folder]["mask"] = mask

    return fi_dict


def furnitureItemFeatures(hero: np.array, fi_dict: dict,
                          lvlRatioDict: dict = None):
    """
    Runs template matching FI identification against the 'hero' passed in.
        When lvlRatioDict is passed in the templates will be rescaled to
        attempt and find the best template size for detecting FI objects

    Args:
        hero: np.array(x,y.3) representing an rgb image
        fi_dict: dictionary of information about each FI template to get ran
            against the image
        lvlRatioDict: dictionary that contains the predicted height of each
            signature item based on precomputed text to si scaling calculations
    Returns:
        dictionary with best "score" that each template achieved on the 'hero'
            image
    """
    # variable_multiplier =
    x, y, _ = hero.shape
    x_div = 2.4
    y_div = 2.0
    x_offset = int(x*0.1)
    y_offset = int(y*0.30)
    # hero_copy = hero.copy()

    crop_hero = hero[y_offset: int(y*0.6), x_offset: int(x*0.4)]

    numberScore = {}
    neighborhood_size = 7
    sigmaColor = sigmaSpace = 75.

    # size_multiplier = 4

    old_crop_hero = crop_hero
    crop_hero = cv2.bilateralFilter(
        crop_hero, neighborhood_size, sigmaColor, sigmaSpace)

    # rgb_range = [
    #     np.array([190, 34, 0]), np.array([255, 184, 157])]
    # rgb_range = [
    #     np.array([180, 0, 0]), np.array([255, 246, 255])]

    # default_blur = processing.blur_image(
    #     crop_hero, rgb_range=rgb_range)
    # # (RMin = 180 , G
    # hero_crop_contours = cv2.findContours(
    #     default_blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # hero_crop_contours = imutils.grab_contours(hero_crop_contours)

    # # TODO add contour detection instead of template matching for low
    #   resolution images
    # hsv_range = [
    #     np.array([0, 50, 124]), np.array([49, 255, 255])]
# (RMin = 133 , GMin = 61, BMin = 35), (RMax = 255 , GMax = 151, BMax = 120)
    rgb_range = [np.array([133, 61, 35]), np.array([255, 151, 120])]

    blur_hero = processing.blur_image(crop_hero, rgb_range=rgb_range)

    fi_color_contours = cv2.findContours(
        blur_hero, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    fi_color_contours = imutils.grab_contours(fi_color_contours)
    fi_color_contours = sorted(fi_color_contours, key=cv2.contourArea,
                               reverse=True)[:2]
    # # # for fi_color_contour in fi_color_contour:
    # blur_hero_mask = np.zeros_like(blur_hero)
    crop_hero_mask = np.zeros_like(crop_hero)
    # hsv_rgb_mask = np.zeros_like(blur_hero)

    # master_contour = [
    #     _cont for _cont_list in fi_color_contours for _cont in _cont_list]
    # hull = cv2.convexHull(np.array(master_contour))

    # cv2.drawContours(crop_hero_mask, [
    #     hull], -1, (255, 255, 255), thickness=cv2.FILLED)

    for _cont in fi_color_contours:
        hull = cv2.convexHull(np.array(_cont))
        cv2.drawContours(crop_hero_mask, [
            hull], -1, (255, 255, 255), thickness=cv2.FILLED)
        # cv2.drawContours(crop_hero_mask, [_cont], -1, (255, 0, 0))

    # cv2.drawContours(blur_hero_mask, fi_color_contours, -1,
    #                  (255, 255, 255))
    # #  , thickness=cv2.FILLED)
    # # cv2.drawContours(crop_hero_mask, hero_crop_contours, -1,
    # #                  (255, 255, 255))
    # #  , thickness=cv2.FILLED)

    # cv2.drawContours(hsv_rgb_mask, fi_color_contours, -1,
    #                  (255, 255, 255), thickness=cv2.FILLED)
    # cv2.drawContours(hsv_rgb_mask, hero_crop_contours, -1,
    #                  (255, 255, 255), thickness=cv2.FILLED)

    # # color_mask = cv2.merge(
    # #     [blur_hero, blur_hero, blur_hero])
    blur_hero = cv2.bitwise_and(crop_hero_mask, crop_hero)
    blur_hero[np.where((blur_hero == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    old_crop_hero = cv2.bitwise_and(crop_hero_mask, old_crop_hero)

    # blur_hero = crop_hero
    for pixel_offset in range(-5, 15, 1):

        for folder_name, imageDict in fi_dict.items():
            fi_image = imageDict["template"]

            sourceSIImage = imageDict["source"]
            hero_h, hero_w = sourceSIImage.shape[:2]

            original_height, original_width = fi_image.shape[:2]

            base_height_ratio = original_height/hero_h
            # resize_height
            base_new_height = max(round(
                lvlRatioDict[folder_name]["height"]), 12)+pixel_offset
            # base_new_height = round(lvlRatioDict[folder_name]["height"])

            new_height = round(base_new_height * base_height_ratio)
            scale_ratio = new_height/original_height
            new_width = round(original_width * scale_ratio)
            fi_image = cv2.resize(
                fi_image, (new_width, new_height))
            # fi_gray = cv2.cvtColor(fi_image, cv2.COLOR_BGR2GRAY)

            #   Min = 0, BMin = 0), (RMax = 255 , GMax = 246, BMax = 255)

            mask = cv2.resize(
                imageDict["mask"], (new_width, new_height))
            # mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # hero_gray = cv2.cvtColor(crop_hero, cv2.COLOR_BGR2GRAY)

            height, width = fi_image.shape[:2]

            # sizedROI = cv2.resize(
            #     hero, (int(x * image_ratio), int(y * image_ratio)))
            # if folder_name != "0":
            #     mask_gray = cv2.bitwise_not(mask_gray)

            try:
                templateMatch = cv2.matchTemplate(
                    blur_hero, fi_image, cv2.TM_CCOEFF_NORMED,
                    mask=mask)
            except Exception:
                if blur_hero.shape[0] < fi_image.shape[0] or \
                        blur_hero.shape[1] < fi_image.shape[1]:
                    _height, _width = fi_image.shape[:2]
                    blur_hero = hero[
                        y_offset: max(int(y/y_div),
                                      int(_height * 1.2)+y_offset),
                        x_offset: max(int(x/x_div),
                                      int(_width * 1.2)+x_offset), ]
                    # blur_hero = crop_hero
                    # hero_gray = cv2.cvtColor(crop_hero, cv2.COLOR_BGR2GRAY)
                templateMatch = cv2.matchTemplate(
                    blur_hero, fi_image, cv2.TM_CCOEFF_NORMED,
                    mask=mask)

            (_, score, _, scoreLoc) = cv2.minMaxLoc(templateMatch)
            scoreLoc = (scoreLoc[0], scoreLoc[1])
            coords = (scoreLoc[0] + width, scoreLoc[1] + height)

            if folder_name not in numberScore:
                numberScore[folder_name] = []
            numberScore[folder_name].append(
                (score, pixel_offset, (scoreLoc, coords)))
    best_score = {}
    import math
    for _folder, _fi_scores in numberScore.items():
        numberScore[_folder] = sorted(_fi_scores, key=lambda x: x[0])
        _t = [_num for _num in numberScore[_folder]
              if not math.isinf(_num[0])]
        if len(_t) == 0:
            if GV.VERBOSE_LEVEL >= 1:
                print("Failed to find FI", _folder,  numberScore[_folder])
            _best_match = (0, 0, ((0, 0), (0, 0)))
        else:
            _best_match = _t[-1]

        _score_loc = _best_match[2][0]
        _coords = _best_match[2][1]
        cv2.rectangle(blur_hero, _score_loc, _coords, (255, 0, 0), 1)
        best_score[_folder] = _best_match[0]

    return best_score


def digitFeatures(digit: np.array, saveDir=None):
    """
    Save a presized digit to whatever number is entered
    Args:
        digit: presized image of a digit, that will be saved as a training
            template under whatever digitName/label is entered when prompted
            by terminal
    Return:
        None
    """

    baseDir = GV.staminaTemplatesPath
    if saveDir:
        baseDir = saveDir
    digitFolders = os.listdir(baseDir)
    plt.figure()
    plt.imshow(digit)
    plt.ion()

    plt.show()

    number = input("Please enter the number shown in the image: ")

    plt.close()

    if number not in digitFolders:
        print("No such folder {}".format(number))
        number = "none"

    numberDir = os.path.join(baseDir, number)
    numberLen = str(len(os.listdir(numberDir)))
    numberName = os.path.join(numberDir, numberLen)

    cv2.imwrite("{}.png".format(numberName), digit)


if __name__ == "__main__":

    # Load in base truth/reference images
    files = load.findFiles("../hero_icon/*")
    baseImages = []
    for i in files:
        hero = cv2.imread(i)
        baseName = os.path.basename(i)
        name, ext = os.path.splitext(baseName)

        baseImages.append((name, hero))

    # load in screenshot of heroes
    stamina_image = cv2.imread("./stamina.jpg")
    heroesDict, rows = processing.getHeroes(stamina_image)

    cropHeroes = load.crop_heroes(heroesDict)
    imageDB = load.build_flann(baseImages)

    for k, v in cropHeroes.items():

        name, baseHeroImage = imageDB.search(v, display=False)
        heroesDict[k]["label"] = name

    staminaAreas = get_stamina_area(rows, heroesDict, stamina_image)
    staminaOutput = get_text(staminaAreas)
    output = {}

    for name, text in staminaOutput.items():
        label = heroesDict[name]["label"]
        if label not in output:
            output[label] = {}
        output[label]["stamina"] = text

    outputJson = json.dumps(output)
