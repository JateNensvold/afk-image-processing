from typing import List
from numpy import int32
from numpy.typing import NDArray

POINT = NDArray[int32]
POINT_WRAPPER = NDArray[POINT]
CONTOUR = NDArray[POINT_WRAPPER]
CONTOUR_LIST = List[CONTOUR]
