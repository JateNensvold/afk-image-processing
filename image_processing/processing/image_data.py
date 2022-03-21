from numpy import ndarray

from image_processing.afk.roster.RowItem import RowItem


class SegmentResult:
    """
    class with info describing the location of subimage in a larger image 

        Ex. A Hero Portrait from a Hero Roster in AFK Arena
    """

    def __init__(self, segment_name: str, segment_image: ndarray,
                 segment_location: RowItem):
        """_summary_

        Args:
            segment_name (str): _description_
            segment_image (np.ndarray): _description_
            segment_location (RI.RowItem): _description_
        """
        self.name = segment_name
        self.image = segment_image
        self.segment_location = segment_location
