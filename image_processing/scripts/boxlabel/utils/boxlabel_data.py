import collections
import requests
import cv2
from io import BytesIO
from typing import Dict, List, NamedTuple, Set

import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

from image_processing.scripts.boxlabel.utils.input_arguments import (
    BOX_LABEL_DATA_FORMAT)


# Example of Boxlabel image instance
# {
#     "ID": "ckrzb0yve5z660yc06dkjbm9s",
#     "DataRow ID": "ckrzaovk95jj10ypg903y8ieu",
#     "Labeled Data": "https://storage.labelbox.com/ckrz9srr563510ydl0bpv94w1%2Fc6e656aa-d647-b45e-4cd8-dc38f99dc4e5-brutus_000_0_jpg.rf.3e65ea8237b620c76d8bafd20635bf01.jpg?Expires=1648283322195&KeyName=labelbox-assets-key-3&Signature=QaLASJkOSCPFo46yRoOXsw1Ylus",
#     "Label": {
#         "objects": [
#             {
#                 "featureId": "ckrzb125j00003b66pyvdb6r5",
#                 "schemaId": "ckrzb0v9j68og0ydlh9at59b4",
#                 "color": "#006FA6",
#                 "title": "Mythic",
#                 "value": "mythic",
#                 "instanceURI": "https://api.labelbox.com/masks/feature/ckrzb125j00003b66pyvdb6r5?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3J6OXNycmk2MzUyMHlkbDhnMDI4OHA4Iiwib3JnYW5pemF0aW9uSWQiOiJja3J6OXNycjU2MzUxMHlkbDBicHY5NHcxIiwiaWF0IjoxNjQ3MDczNzIxLCJleHAiOjE2NDk2NjU3MjF9.S37F6Dfh5To4tdCtXFd7az1A0lFTq6miMFBF-FeN2Z8"
#             }
#         ],
#         "classifications": [],
#         "relationships": []
#     },
#     "Created By": "xxenderwarriorxx@gmail.com",
#     "Project Name": "Ascension Border",
#     "Created At": "2021-08-05T19:21:30.000Z",
#     "Updated At": "2021-08-05T19:21:30.406Z",
#     "Seconds to Label": 0.257,
#     "External ID": "brutus_000_0_jpg.rf.3e65ea8237b620c76d8bafd20635bf01.jpg",
#     "Agreement": -1,
#     "Benchmark Agreement": -1,
#     "Benchmark ID": null,
#     "Dataset Name": "Ascension Border",
#     "Reviews": [],
#     "View Label": "https://editor.labelbox.com?project=ckrzarwja0y510y9u4dc009id&label=ckrzb0yve5z660yc06dkjbm9s",
#     "Has Open Issues": 0,
#     "Skipped": false
# },


class BoundingBox(NamedTuple):
    """
    A data class to hold the information about bounding box data
    """
    x_coord: int
    y_coord: int
    width: int
    height: int


class SegmentationData(NamedTuple):
    """
    A data class to hold information about image segmentation data
    """
    segmentation_class_index: str
    segmentation: List[int]
    bounding_box: BoundingBox
    area: int


class ClassData(NamedTuple):
    """
    Data from a Box Label Json in an organize format
    """

    class_names: List[str]
    class_distribution: Dict[str, BOX_LABEL_DATA_FORMAT]


def download_image(image_url: str):
    """
    Download image into memory from image_url

    Args:
        image_url (str): location to download image from

    Returns:
        _type_: _description_
    """
    image = np.array(
        Image.open(BytesIO(requests.get(
            image_url).content)))
    return image


def get_annotation(class_name: str, mask: np.ndarray):
    '''
    Takes single binary image 'mask' and gets the segmentation, bounding box
        and area of the white part of the image
    Args:
        class_name (str): name of segmentation class
        mask(np.ndarray): binary image stored as np.ndarray
    Return:
        SegmentationData with 
    '''

    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation: List[int] = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    polygons = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
    run_length_encoding = cocomask.merge(polygons)
    area = cocomask.area(run_length_encoding)
    bounding_box = BoundingBox(*cv2.boundingRect(mask))

    # find contours should only find a singular contour so segmentation list
    #   should only have a single element in it, resulting in only the first
    #   element getting passed to SegmentationData
    segmentation_data = SegmentationData(
        class_name, segmentation[0], bounding_box, area)
    return segmentation_data


def get_classes(json_data: BOX_LABEL_DATA_FORMAT):
    """
    Get list of all classes in json_data

    Args:
        json_data (BOX_LABEL_DATA_FORMAT): labelbox json data
    Returns:
        List[str] : list of all unique class names found in json_data 
    """
    class_set: Set[str] = set()
    class_frequency: Dict[str,
                          BOX_LABEL_DATA_FORMAT] = collections.defaultdict(list)
    for image_label_data in json_data:
        for label_instance in image_label_data["Label"]["objects"]:
            class_name = label_instance["value"]
            class_set.add(class_name)
        class_frequency[class_name].append(image_label_data)

    return ClassData(list(class_set), class_frequency)
