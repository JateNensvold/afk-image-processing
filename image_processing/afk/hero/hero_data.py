import json
from typing import List
import numpy as np

import image_processing.models.model_attributes as MA
from image_processing.afk.roster.matrix import Matrix


class HeroImage:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, hero_name: str, image: np.ndarray):
        """_summary_

        Args:
            hero_name (str): _description_
            image (np.ndarray): _description_

        Returns:
            _type_: _description_
        """

        self.name = hero_name
        self.image = image


class RosterData:
    """_summary_
    """

    def __init__(self, hero_data_list: List["DetectedHeroData"],
                 roster_matrix: Matrix):
        """_summary_

        Returns:
            _type_: _description_
        """
        self.hero_data_list = hero_data_list
        self.roster_matrix = roster_matrix

    def json(self):
        """_summary_
        """
        json_dict = {}

        json_dict["rows"] = len(self.roster_matrix)
        json_dict["columns"] = len(max(self.roster_matrix, key=len))
        json_dict["heroes"] = []
        for hero_data in self.hero_data_list:
            json_dict["heroes"].append(str(hero_data))

        return json_dict


class DetectedHeroData:
    """_summary_
    """

    def __init__(self, hero_name: str, signature_item: MA.ModelResult,
                 furniture: MA.ModelResult, hero_ascension: MA.ModelResult,
                 hero_engraving: MA.ModelResult, image: np.ndarray = None):
        """_summary_

        Args:
            hero_name (str): _description_
            signature_item (MA.ModelResult): _description_
            furniture (MA.ModelResult): _description_
            hero_ascension (MA.ModelResult): _description_
            hero_engraving (MA.ModelResult): _description_
            image (np.ndarray, optional): _description_. Defaults to None.
        """
        self.name = hero_name
        self.signature_item = signature_item
        self.furniture = furniture
        self.ascension = hero_ascension
        self.engraving = hero_engraving
        self.image = image

    def __str__(self):
        """_summary_
        """
        hero_dict = {
            "name": self.name,
            "signature_item": self.signature_item.label,
            "furniture": self.furniture.label,
            "ascension": self.ascension.label,
            "engraving": self.engraving.label
        }
        return json.dumps(hero_dict)

    def __repr__(self) -> str:
        return str(self)