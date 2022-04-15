import os
import argparse

from image_processing.scripts.boxlabel.utils.boxlabel_data import get_classes
from image_processing.scripts.boxlabel.utils.input_arguments import load_json


def fetch_classes():
    """
    Return classes from labelbox file

    Returns:
        List[str]: list of classes in labelbox file
    """
    parser = argparse.ArgumentParser(
        description="Fetch and print classes from labelbox file")
    parser.add_argument("box_label_file", type=str,
                        help="Path to box Label JSON file")
    args = parser.parse_args()
    box_label_file_path = os.path.abspath(args.box_label_file)
    json_data = load_json(box_label_file_path)
    class_data = get_classes(json_data)
    return class_data.class_names


if __name__ == "__main__":
    class_list = fetch_classes()
    print(
        f"Set of classes used in labelbox file, in no particular order: {class_list}")
