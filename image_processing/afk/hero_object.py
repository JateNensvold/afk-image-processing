import numpy as np


class hero_object:
    def __init__(self, name: str, faction: str = None,
                 image: np.ndarray = None):
        """
        A hero object used to store attributes about AFK arena heroes that are
            manipulated throughout image_processing

        Args:
            name: name of hero
            faction: faction of hero
            image: source image of hero
        """
        self.name = name
        if faction is not None:
            self.faction = faction
        if image is not None:
            self.image = image
