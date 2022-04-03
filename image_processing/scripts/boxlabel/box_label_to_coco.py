import time
import json
import os
import pathlib
import datetime

from typing import Dict, Any

import cv2
import numpy as np

from image_processing.scripts.boxlabel.utils.input_arguments import (
    parse_args, load_json)
from image_processing.scripts.boxlabel.utils.boxlabel_data import (
    download_image, get_annotation)


INFO = {
    "description": "AFK Arena Hero Info Dataset",
    "url": "https://github.com/JateNensvold/afk-image-processing",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "jatenensvold",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = {
    "border": {
        'id': 0,
        'name': 'border',
        'supercategory': 'none',
    },
    "elite": {
        'id': 1,
        'name': 'elite',
        'supercategory': 'border',
    },
    "elite+": {
        'id': 2,
        'name': 'elite+',
        'supercategory': 'border',
    },
    "legendary": {
        'id': 3,
        'name': 'legendary',
        'supercategory': 'border',
    },
    "legendary+": {
        'id': 4,
        'name': 'legendary+',
        'supercategory': 'border',
    },
    "mythic": {
        'id': 5,
        'name': 'mythic',
        'supercategory': 'border',
    },
    "mythic+": {
        'id': 6,
        'name': 'mythic+',
        'supercategory': 'border',
    },
    "ascended": {
        'id': 7,
        'name': 'ascended',
        'supercategory': 'border',
    },
}


def visualize_mask(image: np.ndarray,
                   tool: Dict[str, Any],
                   alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a mask onto an image

    Args:
        image (np.ndarray): image to overlay mask onto
        tool (Dict[str,any]): Dict response from the export
        alpha: How much to weight the mask when adding to the image
    Returns:
        image with a point drawn on it.
    """
    raw_image_mask = download_image(tool["instanceURI"])
    raw_image_mask[:, :, 1] *= 0
    raw_image_mask[:, :, 2] *= 0
    weighted = cv2.addWeighted(image, alpha, raw_image_mask, 1 - alpha, 0)
    image[np.sum(raw_image_mask, axis=-1) >
          0] = weighted[np.sum(raw_image_mask, axis=-1) > 0]

    return image


def convert_format(input_data: dict, output_dir: str = "./", license_index: int = 0,
                   type_label: str = "instances", _split=(98, 1, 1)):
    '''
    Converts https://labelbox.com/ instance segmentation JSON file to COCO
        Folder/JSON format
    Creates COCO output in 'output_dir' when one is provided otherwise
        defaults to generating the output in the directory this file is ran
        from.
    Args:
        input_data: dictionary of input JSON data, loaded from file using
            json.loads
        output_dir: location to create COCO output at
        license_index: index of license to be used from global variable
            LICENSES
        type_label: label for JSON output that gets written to disk, doesn't
            actually change anything besides label tag in the top level JSON
            dict

    '''
    output_dir = os.path.abspath(output_dir)
    output_dir_files = os.listdir(output_dir)
    coco_dir_count = 0
    for _file_name in output_dir_files:
        if "coco_data" in _file_name:
            coco_dir_count += 1
    coco_dir = os.path.join(output_dir, "coco_data" + str(coco_dir_count))

    pathlib.Path(coco_dir).mkdir(parents=True, exist_ok=True)
    coco_train = os.path.join(coco_dir, "train")
    coco_test = os.path.join(coco_dir, "test")
    coco_valid = os.path.join(coco_dir, "valid")
    pathlib.Path(coco_train).mkdir(parents=True, exist_ok=True)
    pathlib.Path(coco_test).mkdir(parents=True, exist_ok=True)
    pathlib.Path(coco_valid).mkdir(parents=True, exist_ok=True)

    image_data_list = []
    annotation_data_list = []

    annotation_id = 0
    for image_id, segmentation_image in enumerate(input_data):
        start_time = time.time()
        # Image Data
        temp_image_data_dict = {}
        image_name = segmentation_image["External ID"]
        print(f"Starting:{image_id} {image_name} ...")
        image_path = os.path.join(coco_train, image_name)
        image = download_image(segmentation_image["Labeled Data"])
        height, width = image.shape[:2]
        cv2.imwrite(image_path, image)
        temp_image_data_dict["id"] = image_id
        temp_image_data_dict["license"] = LICENSES[license_index]["id"]
        temp_image_data_dict["file_name"] = image_name
        temp_image_data_dict["height"] = height
        temp_image_data_dict["width"] = width
        temp_image_data_dict["date_captured"] = segmentation_image[
            "Updated At"]
        image_data_list.append(temp_image_data_dict)

        # Annotation Data
        for segmentation_object in segmentation_image["Label"]["objects"]:
            segmentation_class_name = segmentation_object["value"]

            temp_annotation_data_dict = {}
            temp_annotation_data_dict["id"] = annotation_id
            temp_annotation_data_dict["image_id"] = image_id
            temp_annotation_data_dict["category_id"] = CATEGORIES[
                segmentation_class_name]["id"]
            raw_image_mask = download_image(segmentation_object["instanceURI"])
            gray_image_mask = cv2.cvtColor(raw_image_mask, cv2.COLOR_BGR2GRAY)

            segmentation_data = get_annotation(
                segmentation_class_name, gray_image_mask)
            temp_annotation_data_dict["bbox"] = segmentation_data.bounding_box
            temp_annotation_data_dict["area"] = float(segmentation_data.area)
            temp_annotation_data_dict["segmentation"] = segmentation_data.segmentation
            temp_annotation_data_dict["iscrowd"] = 0

            annotation_id += 1
        finish_time = time.time()
        annotation_data_list.append(temp_annotation_data_dict)
        print(
            f"Finished:{image_id} {image_name} in {finish_time - start_time}")

    coco_train_json = os.path.join(coco_train, "coco.json")

    json_data = {"info": INFO,
                 "images": image_data_list,
                 "licenses": LICENSES,
                 "type": type_label,
                 "annotations": annotation_data_list,
                 "categories": [category for _name, category
                                in CATEGORIES.items()]}

    with open(coco_train_json, "w", encoding="utf-8") as coco_json_file:
        json_dump = json.dumps(json_data)

        coco_json_file.write(json_dump)


def generate_coco(output_dir: str, json_path: str):
    """
    Generate and write data to Files following the COCO data format
    https://cocodataset.org/#home

    Args:
        output_dir (str): directory that the output data direcotry will get
            generated in
        json_path (str): path to json file that contains the labelbox data
    """
    json_data = load_json(json_path)
    convert_format(json_data, output_dir=output_dir)


if __name__ == "__main__":

    args = parse_args()

    generate_coco(args.output_dir, args.box_label_file)
