import argparse
import json
from typing import Any, Dict, List

BOX_LABEL_INSTANCE = Dict[str, Any]
BOX_LABEL_DATA_FORMAT = List[BOX_LABEL_INSTANCE]


def parse_args():
    """
    Parse input args for converting labelbox formated results into YolOV5/COCO data format

    Returns:
       namespace : argparse namespace containing parse results
    """
    parser = argparse.ArgumentParser(
        description="Convert labelbox file into COCO/YoloV5 formated files")
    parser.add_argument("box_label_file", type=str,
                        help="Path to box Label JSON file")
    parser.add_argument(
        "output_dir", type=str,
        help="Path to directory to create COCO/YoloV5 output directory at")

    args = parser.parse_args()
    return args


def load_json(json_path: str):
    """_summary_

    Args:
        json_path (str): path to json_file
    Returns:
        BOX_LABEL_DATA_FORMAT : labelbox json data loaded into python dictionary
    """
    box_label_data: BOX_LABEL_DATA_FORMAT = {}

    with open(json_path, "r", encoding="utf-8") as json_file:
        box_label_data = json.loads(json_file.read())

    return box_label_data
