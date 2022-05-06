
from typing import Any, TypeVar, Union, NamedTuple, MutableSequence


S = TypeVar("S")
T = TypeVar("T")


class DictTuple(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_

    Returns:
        _type_: _description_
    """
    key: Any
    value: Any


class UtilList(list, MutableSequence[S]):
    """_summary_

    Args:
        list (_type_): _description_
    """

    def first(self, default: T = None) -> Union[S, T]:
        """_summary_

        Returns:
            _type_: _description_
        """
        if len(self) == 0:
            return default
        else:
            return self[0]
