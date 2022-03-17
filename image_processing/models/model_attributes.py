"""
This Module is used to MAP pytorch model's numerical class labels onto
feature labels
"""
from typing import Dict, NamedTuple, Union


_SI_FI_MODEL_LABELS = ['0 si', '1 star', '10SI', '2 star', '20 si',
                       '3 fi', '3 star', '30 si', '4 star', '5 star', '9 fi']
BORDER_MODEL_LABELS = ["B", "E", "E+", "L", "L+", "M", "M+", "A"]

SI_LABELS = {
    0: "0",
    2: "10",
    4: "20",
    7: "30"
}

FI_LABELS = {
    5: "3",
    10: "9"
}

ASCENSION_STAR_LABELS = {
    1: "A1",
    3: "A2",
    6: "A3",
    8: "A4",
    9: "A5"
}


class ModelResult(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """
    label: str
    score: int

    @classmethod
    def from_dict(cls, model_result: Dict[str, Union[str, int]]):
        """_summary_

        Args:
            model_result (Dict[str, str]): _description_
        """

        return ModelResult(model_result["label"], model_result["score"])

    def to_dict(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {"label": self.label, "score": self.score}
