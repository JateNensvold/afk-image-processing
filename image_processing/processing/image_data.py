from numpy import ndarray

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from image_processing.afk.roster.RowItem import RowItem


class SegmentResult:
    """
    class with info describing the location of subimage in a larger image

        Ex. A Hero Portrait from a Hero Roster in AFK Arena
    """

    def __init__(self, segment_name: str, segment_image: ndarray,
                 segment_location: "RowItem", source_image: ndarray):
        """_summary_

        Args:
            segment_name (str): name/identifier for segment
            segment_image (np.ndarray): image area described by
                segment_location(possibly resized, no guarantee to be same width/height)
            segment_location (RI.RowItem): location of segment in larger image
            source_image (np.ndarray): source image that segment_location
                describes a portion of
        """
        self.name = segment_name
        self.image = segment_image
        self.segment_location = segment_location
        self.source_image = source_image
