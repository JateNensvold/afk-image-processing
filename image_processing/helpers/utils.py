from typing import TypeVar, List
from numpy import median

T = TypeVar("T")  # pylint: disable=invalid-name


def list_median(list_item: List[T]) -> T:
    """_summary_
    Sort a list and return the median value in it

    Args:
        list_item (List[Any]): list to find median of
    """
    list_item.sort()

    return median(list_item)
