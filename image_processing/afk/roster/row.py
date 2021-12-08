"""
A module containing all functionality that is related to the Row Class

The Row Class provides a wrapper around a list of RowItems so they can be
interfaces like they were in a normal list while also benifiting from some
extended functionality such as RowItem lookup by key values and other helper
methods to interact with grouped RowItems
"""
from typing import TYPE_CHECKING

import rtree
import numpy as np

import image_processing.globals as GV
import image_processing.afk.roster.RowItem as RI


if TYPE_CHECKING:
    import image_processing.afk.roster.column_objects as CO


class RowIntersectionException(Exception):
    """
    Error to be raised when two RowItems have a collision
    """


class Row():
    """
    A class wrapper for a 1 dimensional series of objects

    Raises:
        StopIteration: When __next__ is called and the end of the series has
            been reached
        KeyError: When a
        RowIntersectionException: [description]

    Returns:
        [type]: [description]
    """
    # pylint: disable=protected-access

    def __init__(self, columns: "CO.ColumnObjects"):
        """
        Create Row used to hold objects that have dimensions

        Args:
            columns: columnsObject used to calculate what column each
                RI.RowItem is in
        """
        self._row_items_by_name: dict[str, RI.RowItem] = {}
        self._row_items_by_id: dict[str, RI.RowItem] = {}

        self._row_items: list[RI.RowItem] = []
        self._idx = 0
        self.rtree = rtree.index.Index()
        self.columns = columns
        self.head: int = None
        self.avg_width = 0

    def __str__(self):
        return (
            f"({self.head} "
            f"({''.join([str(_row_item) for _row_item in self._row_items])}))")

    def __iter__(self):
        return self

    def __next__(self) -> RI.RowItem:
        """
        Iterate over all RowItems in row
        """
        self._idx += 1
        try:
            return self._row_items[self._idx - 1]
        except IndexError:
            self._idx = 0
            raise StopIteration  # pylint: disable=raise-missing-from

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

    def get(self, name: str, id_lookup=False):
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
        except KeyError as exception_handle:
            if not id_lookup:
                for _row_item in self._row_items:
                    if name in _row_item.alias:
                        return self._row_items_by_name[_row_item.name]
            raise KeyError(f"Key '{name}' not found in row "
                           f"'{self.get_head()}'") from exception_handle

    def __getitem__(self, index: int) -> RI.RowItem:
        """
        Get an item by its index
        Args:
            index: position of item in row
        Returns:
            import image_processing.afk.roster.RowItem
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
        output = [_row_item.dimensions.height for _row_item in self._row_items]
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

    def get_average(self) -> float:
        """
        Get average width of all RowItems stored within Row

        Returns:
            [float]: average width of RowItems
        """
        width_list = [
            _row_item.dimensions.width for _row_item in self._row_items]
        return np.mean(width_list)

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
            dimensions: x,y,width,height of object
            name: identifier for row item, can be used for lookup later
            detect_collision: check for object overlap/collisions when
                appending to row
        Return:
            Index(int) RowItem was added at when successfully appended

            otherwise returns -1(int)
        """
        collision_status = -1
        temp_row_item = RI.RowItem(dimensions, name)
        if len(self._row_items) == 0:
            self.head = temp_row_item.dimensions.y
        if detect_collision:
            collision_status = self.check_collision(temp_row_item)

            # If collision is successful, don't return\
            if collision_status != -1:
                # output = temp_row_item.dimensions.overlap(
                #     self._row_items_by_id[collision_status].dimensions)
                # print("collision", collision_status,
                #       output/temp_row_item.dimensions.size())
                return collision_status
        temp_id = id(temp_row_item)
        self._row_items_by_name[name] = temp_row_item
        self._row_items_by_id[temp_id] = temp_row_item

        # avg_width = self._add_average(_temp_RowItem.dimensions.width)
        # self.avg_width = avg_width
        self._row_items.append(temp_row_item)

        self.rtree.insert(id(temp_row_item),
                          temp_row_item.dimensions.coords())
        self.columns.find_column(temp_row_item)
        return collision_status

    def check_collision(self, new_row_item: RI.RowItem,
                        resolve_error: bool = True,
                        collision_overlap=0.75,
                        **dimension_object_kwargs):
        """
        Check if row_item's dimensions overlap with any of the objects
            in the row object, and merge collision_object with overlaping
            object if collisions is detected
        Args:
            row_item: new RowItem to check against existing RowItems
            resolve_error: flag to resolve errors when multi RowItem collisions
                occur and return the item with the greatest overlap
            collision_overlap: if the collision is any less than this number
                return the collision index but don't merge
        Return:
            When collision occurs id(int) of updated row object is returned

            otherwise -1(int) is returned
        """

        _new_item_coords = new_row_item.dimensions.coords()
        _intersections_list = list(self.rtree.intersection(_new_item_coords))
        if len(_intersections_list) == 1:
            _intersection_id = _intersections_list[0]

            return self._merge(_intersection_id, new_row_item,
                               collision_overlap, **dimension_object_kwargs)
        elif len(_intersections_list) > 1:
            _intersection_objects = [
                self._row_items_by_id[_intersection_id]
                for _intersection_id in _intersections_list]
            # Attempt to merge with largest collision object
            if resolve_error:
                overlap_list: tuple[int, int] = []
                for _row_item in _intersection_objects:
                    overlap_tuple = _row_item.dimensions._overlap_percent(
                        new_row_item.dimensions)
                    overlap_list.append((id(_row_item), overlap_tuple))
                # print(overlap_list)
                max_overlap_tuple = max(
                    overlap_list, key=lambda overlap_tuple: overlap_tuple[1])
                return self._merge(max_overlap_tuple[0], new_row_item,
                                   collision_overlap,
                                   **dimension_object_kwargs)

            raise RowIntersectionException(
                "More than one intersection occurred "
                f"between {new_row_item} and "
                f"{[str(_i) for _i in _intersection_objects]}")
        else:
            # No intersection at all
            return -1

    def _merge(self, collision_item_id: int, new_row_item: RI.RowItem,
               collision_overlap: int,
               **dimension_object_kwargs):
        """
            Merge RowItem that corresponds with 'collision_item_id' 'and
            new_row_item' when collision_overlap threshold is met
        Args:
            collision_item_id: id of RowItem in row
            new_row_item: RowItem to merge with 'RowItem' corresponding with
                'collision_item_id'
            collision_overlap: if the collision is any less than this number
                return the collision index but don't merge
        Return:
            'collision_item_id'(int)
        """
        _collision_row_item = self._row_items_by_id[collision_item_id]
        _collision_item_coordinates = _collision_row_item.\
            dimensions.coords()

        old_coords = _collision_row_item.dimensions.coords()
        # old_width = _collision_row_item.dimensions.width

        _collision_tuple = _collision_row_item.dimensions._overlap_percent(
            new_row_item.dimensions)
        if _collision_tuple[2] < collision_overlap:
            if GV.verbosity(1):
                print(f"Collision overlap({_collision_tuple[2]}) was below "
                      f"collision threshold({collision_overlap}), returning "
                      "collision ID with no update")
            return collision_item_id

        _collision_row_item.merge(new_row_item, **dimension_object_kwargs)

        # new_width = _collision_row_item.dimensions.width
        if _collision_row_item.dimensions.coords() != old_coords:
            self.rtree.delete(collision_item_id,
                              _collision_item_coordinates)
            # if old_width != new_width:
            #     removed_width = self._remove_average(old_width)
            #     self.avg_width = removed_width
            #     removed_width = self._add_average(new_width)
            #     self.avg_width = removed_width
            self.rtree.insert(collision_item_id,
                              _collision_row_item.dimensions.coords())
        return collision_item_id

    def sort(self):
        '''
        Sort row by x coordinate of each RowItem
        '''
        self._row_items.sort(key=lambda _row_item: _row_item.dimensions.x)
