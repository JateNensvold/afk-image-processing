import json
from typing import List, NamedTuple

from numpy import asarray, array, float32, int64
from faiss import index_factory
from faiss.swigfaiss import IndexIDMap

from image_processing.models.model_attributes import ModelResult

ENGRAVING_DIMENSIONS = 3
INDEX_KEY = "IDMap,Flat"


class EngravingData(NamedTuple):
    """
    Class representing just the color of an engraving
    """
    hue: float
    satuartion: float
    value: float


class EngravingResult(NamedTuple):
    """
    Data class that represents the HSV values of a hero engraving
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
        return EngravingData(self.hue, self.saturation, self.value)


class EngravingSearch:
    """_summary_
    """

    def __init__(self, engraving_data_list: List[EngravingResult] = None):
        """_summary_
        """
        self.index: IndexIDMap = index_factory(ENGRAVING_DIMENSIONS, INDEX_KEY)
        for engraving_result in engraving_data_list:
            self.index.add_with_ids(asarray([engraving_result.data()], dtype=float32),
                                    array([engraving_result.name], int64))

    @classmethod
    def from_json(cls, engraving_json_path: str):
        """_summary_

        Args:
            json_data (Dict[str]): _description_
        """
        engraving_list: List[EngravingResult] = []
        with open(engraving_json_path, 'r', encoding="utf-8") as file:
            engraving_json_data = json.load(file)
            for engraving_name, engraving_data in engraving_json_data["engravings"].items():
                engraving_list.append(EngravingResult(
                    engraving_name,
                    engraving_data["hue"],
                    engraving_data["saturation"],
                    engraving_data["value"]))
        return EngravingSearch(engraving_list)

    def search(self, engraving_data: EngravingData):
        """_summary_

        Args:
            engraving_index (Tuple[float32]): _description_
        """
        distance_array, id_array = self.index.search(
            asarray([engraving_data], dtype=float32), k=1)
        result_name = str(id_array[0][0])
        distance_value = float(distance_array[0][0])
        output = ModelResult(result_name, distance_value)
        return output
