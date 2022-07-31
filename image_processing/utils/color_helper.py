from enum import Enum
from typing import Any, Dict, NamedTuple
import matplotlib as mpl
import jsonpickle


class RGBColorNames(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class PickleRGB(jsonpickle.handlers.BaseHandler):
    """_summary_

    Args:
        jsonpickle (_type_): _description_
    """

    def flatten(self, obj: "RGBColor", data: Dict[str, Any]):
        """_summary_

        Args:
            obj (_type_): _description_
            data (_type_): _description_
        """
        data["py/object"] = ".".join((RGBColor.__module__,
                                     RGBColor.__qualname__))

        data[RGBColorNames.RED.value] = int(obj.red)
        data[RGBColorNames.BLUE.value] = int(obj.blue)
        data[RGBColorNames.GREEN.value] = int(obj.green)

        return data

    def restore(self, obj: Dict[str, Any]):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """

        return RGBColor(int(obj[RGBColorNames.RED.value]),
                        int(obj[RGBColorNames.BLUE.value]),
                        int(obj[RGBColorNames.GREEN.value]))


@PickleRGB.handles
class RGBColor(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_

    Returns:
        _type_: _description_
    """
    red: int
    green: int
    blue: int

    @classmethod
    def from_bgr(cls, blue: int, green: int, red: int):
        """_summary_

        Args:
            blue (int): _description_
            green (int): _description_
            red (int): _description_

        Returns:
            _type_: _description_
        """
        return RGBColor(red, green, blue)


class RGBLabel(NamedTuple):
    """
    Data class that represents the Ascension Border of a hero
    """
    name: str
    red: int
    green: int
    blue: int

    def data(self):
        """_summary_

        Returns:
            (AscensionData): return a tuple containing red, green and blue data
        """
        return RGBColor(self.red, self.green, self.blue)


class MplColorHelper:
    """_summary_
    """

    def __init__(self, color_range: int = 255):
        self.color_range = color_range

    def get_rgb(self, color_name: str):
        """_summary_

        Args:
            color_name (str): _description_

        Returns:
            _type_: _description_
        """
        color_tuple = mpl.colors.to_rgba(color_name)
        return [int(color_val*self.color_range) for color_val in color_tuple]

    def get_unicode(self, color_name: str):
        """_summary_

        Args:
            color_name (str): _description_
        """

        return int(mpl.colors.to_hex(color_name)[1:], 16)
