import json
from typing import Dict, List, NamedTuple

from numpy import asarray, array, float32, int64
from faiss import index_factory
from faiss.swigfaiss import IndexIDMap

from image_processing.models.model_attributes import ModelResult

ASCENSION_COLOR_DIMENSIONS = 3
INDEX_KEY = "IDMap,Flat"


class AscensionData(NamedTuple):
    """
    Class representing the HSV color of an Ascension Border
    """
    hue: float
    satuartion: float
    value: float


class AscensionResult(NamedTuple):
    """
    Data class that represents the Ascension Border of a hero
    """
    name: str
    hue: float
    saturation: float
    value: float

    def data(self):
        """_summary_

        Returns:
            (tuple): return a tuple containing hue, saturation and value
        """
        return AscensionData(self.hue, self.saturation, self.value)


class AscensionSearch:
    """_summary_
    """

    def __init__(self, ascension_data_list: List[AscensionResult] = None):
        """_summary_
        """
        self.index: IndexIDMap = index_factory(
            ASCENSION_COLOR_DIMENSIONS, INDEX_KEY)
        self.index_lookup: Dict[int, str] = {}
        for ascension_index, ascension_result in enumerate(ascension_data_list):
            self.index_lookup[ascension_index] = ascension_result.name
            self.index.add_with_ids(asarray([ascension_result.data()], dtype=float32),
                                    array([ascension_index], int64))
        self.ascension_count = len(ascension_data_list)

    @classmethod
    def from_json(cls, ascension_json_path: str):
        """_summary_

        Args:
            json_data (Dict[str]): _description_
        """
        ascension_list: List[AscensionResult] = []
        with open(ascension_json_path, 'r', encoding="utf-8") as file:
            ascension_json_data = json.load(file)
            for ascension_name, ascension_data in ascension_json_data["ascension"].items():
                ascension_list.append(AscensionResult(
                    ascension_name,
                    ascension_data["hue"],
                    ascension_data["saturation"],
                    ascension_data["value"]))
        return AscensionSearch(ascension_list)

    def search(self, ascension_data: AscensionData, k: int=5):
        """_summary_

        Args:
            ascension_data (AscensionData): object representing the HSV color
                of an Ascension Border
            k (int): number of search results to return (capped by length of
                search database)
        """
        model_result_list: List[ModelResult] = []
        distance_array, id_array = self.index.search(
            asarray([ascension_data], dtype=float32), k=min(k, self.ascension_count))
        for num in range(min(k, self.ascension_count)):
            original_result_name = self.index_lookup[id_array[0][num]]
            result_name = original_result_name[0].upper()
            distance_value = float(distance_array[0][num])
            model_result_list.append(ModelResult(result_name, distance_value))
        return model_result_list
