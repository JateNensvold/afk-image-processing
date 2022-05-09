import json
import re
from typing import Dict, List
import jsonpickle
import numpy as np

from image_processing.models.model_attributes import ModelResult
from image_processing.afk.roster.matrix import Matrix


class HeroImage:
    """
    A wrapper around an image that includes the image itself, the image name
        and the location the image was loaded from

    Returns:
        _type_: _description_
    """

    def __init__(self, raw_hero_name: str, image: np.ndarray, image_path: str,
                 clean_name=True):
        """_summary_

        Args:
            raw_hero_name (str): _description_
            image (np.ndarray): _description_
            image_path (str): _description_
            clean_name (bool, optional): _description_. Defaults to True.
        """
        hero_name = raw_hero_name
        if clean_name:
            # Keep file name of awakened heroes so they don't get mixed with normal version
            if raw_hero_name.endswith("aw") or raw_hero_name.endswith("awakened"):
                name_regex_results = re.split(r"(\.)", raw_hero_name)
                hero_name = name_regex_results[0]
            else:
                name_regex_results = re.split(r"(-|_|\.)", raw_hero_name)
                hero_name = name_regex_results[0]
        self.name = hero_name
        self._raw_name = raw_hero_name
        self.image = image
        self.image_path = image_path

    def __str__(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        return f"HeroImage<{self._raw_name, self.image_path}>"


class RosterJson:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, hero_data_list: List["DetectedHeroData"],
                 row_len: int, column_len: int):
        """_summary_

        Args:
            hero_data_list (List[&quot;DetectedHeroData&quot;]): _description_
            row_len (int): _description_
            column_len (int): _description_

        Returns:
            _type_: _description_
        """
        self.hero_data_list = hero_data_list
        self.column_length = column_len
        self.row_length = row_len

    def json_dict(self):
        """_summary_
        """
        json_dict = {}

        json_dict["rows"] = self.row_length
        json_dict["columns"] = self.column_length
        json_dict["heroes"] = []
        for hero_data in self.hero_data_list:
            json_dict["heroes"].append(hero_data.to_dict())

        return json_dict

    def json(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return json.dumps(self.json_dict())

    @classmethod
    def from_json(cls, json_dict: Dict):
        """_summary_

        Args:
            json_dict (Dict): _description_

        Returns:
            _type_: _description_
        """
        raw_hero_list = json_dict["heroes"]
        hero_list = []
        for hero_dict in raw_hero_list:
            detected_hero = DetectedHeroData(
                hero_dict["name"],
                ModelResult.from_dict(hero_dict["signature_item"]),
                ModelResult.from_dict(hero_dict["furniture"]),
                ModelResult.from_dict(hero_dict["ascension"]),
                ModelResult.from_dict(hero_dict["engraving"]))
            hero_list.append(detected_hero)
        roster_instance = RosterJson(
            hero_list, json_dict["rows"], json_dict["columns"])
        return roster_instance


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

    def roster_json(self):
        """_summary_
        """
        row_length = len(self.roster_matrix)
        column_length = len(max(self.roster_matrix, key=len))
        return RosterJson(self.hero_data_list, row_length, column_length)

    def json(self):
        """_summary_
        """
        return self.roster_json().json()


class DetectedHeroData:
    """_summary_
    """

    def __init__(self, hero_name: str, signature_item: ModelResult,
                 furniture: ModelResult, hero_ascension: ModelResult,
                 hero_engraving: ModelResult, image: np.ndarray = None):
        """_summary_

        Args:
            hero_name (str): _description_
            signature_item (ModelResult): _description_
            furniture (ModelResult): _description_
            hero_ascension (ModelResult): _description_
            hero_engraving (ModelResult): _description_
            image (np.ndarray, optional): _description_. Defaults to None.
        """
        self.name = hero_name
        self.signature_item = signature_item
        self.furniture = furniture
        self.ascension = hero_ascension
        self.engraving = hero_engraving
        self.image = image

    def to_dict(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        hero_dict = {
            "name": self.name,
            "signature_item": self.signature_item.to_dict(),
            "furniture": self.furniture.to_dict(),
            "ascension": self.ascension.to_dict(),
            "engraving": self.engraving.to_dict()
        }
        return hero_dict

    def __str__(self):
        """_summary_
        """
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        return str(self)
