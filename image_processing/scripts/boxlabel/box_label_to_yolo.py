import os
import pathlib
import time
from typing import Iterator, List, NamedTuple

import cv2
import numpy as np
from image_processing.scripts.boxlabel.utils.input_arguments import (
    BOX_LABEL_DATA_FORMAT, parse_args, load_json)

from image_processing.scripts.boxlabel.utils.boxlabel_data import (
    SegmentationData, download_image, get_annotation, get_classes)


class YoloV5FilesTemplate(NamedTuple):
    """
    Base class for YoloV5 files that allows classes that inherit from this 
        to override the __new__ method
    """
    training_directory: str
    validation_directory: str
    data_yaml: str


class YoloV5Files(YoloV5FilesTemplate):
    """
    Directories used in yoloV5 data format
    """

    def __new__(cls, parent_dir: str):
        """
        Create paths to files and directories for YoloV5 data format

        Args:
            parent_dir (str): directory that yolov5 files exist in
        """
        training_dir = os.path.join(parent_dir, "train")
        validation_dir = os.path.join(parent_dir, "valid")
        data_yaml = os.path.join(parent_dir, "data.yaml")

        self = super().__new__(cls, training_dir, validation_dir, data_yaml)
        return self

    def create_directories(self):
        """
        Create the training and validation folders for yoloV5 data
        """
        pathlib.Path(self.training_directory).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(self.validation_directory).mkdir(
            parents=True, exist_ok=True)


def generate_yaml(yolo_files: YoloV5Files,
                  class_names: Iterator[str]):
    """
    Create YoloV5 data.yaml

    Args:
        yolo_files (YoloV5Files): Directories used in yoloV5 data format,
            including where to create data.yaml at
        class_names (Iterable[str]): names of classes in model
    """

    # Create data.yaml
    with open(yolo_files.data_yaml, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(
            f"train: {os.path.abspath(yolo_files.training_directory)}\n")
        yaml_file.write(
            f"val: {os.path.abspath(yolo_files.validation_directory)}\n")
        yaml_file.write("\n")
        yaml_file.write(f"nc: {len(class_names)-1}\n")
        yaml_file.write(f"names: {list(class_names)}\n")


def write_yolo_image(image_path: str, image: np.ndarray,
                     segmentation_data_list: List[SegmentationData]):
    """
    Write a yoloV5 image instance and its corresponding text file to
        the filesystem

    Args:
        image_path (str): path to image getting written
        image (np.ndarray): image getting written
        segmentation_data_list (List[SegmentationData]): list of
            segments/labels for the image
    """

    cv2.imwrite(image_path, image)
    file_name, _extension_name = os.path.splitext(image_path)
    text_file_name = f"{file_name}.txt"
    with open(text_file_name, "w", encoding="utf-8") as segmentation_file:
        for segmentation_data in segmentation_data_list:
            annotation_str = ', '.join(
                str(segment) for segment in segmentation_data.segmentation)
            segmentation_file.write(
                f"{segmentation_data.segmentation_class_index} "
                f"{annotation_str} \n")


def convert_format(json_data: BOX_LABEL_DATA_FORMAT, parent_dir: str):
    """
    Convert data from labelbox format to YoloV5 format

    Args:
        json_data (BOX_LABEL_DATA_FORMAT): labelbox json data
        parent_dir (str): directory that will contain the new folder with the
            newly formated data
    """
    output_dir = os.path.abspath(os.path.join(parent_dir, "yolo_data"))

    yolov5_directories = YoloV5Files(output_dir)
    yolov5_directories.create_directories()

    class_name_index = {}
    class_names = get_classes(json_data)
    for class_index, class_name in enumerate(class_names):
        class_name_index[class_name] = class_index

    # TODO: Add multithreading to this so it can download multiple images as
    #   once, speeding up conversion process significantly
    for image_index, image_label_data in enumerate(json_data):
        start_time = time.time()
        image_name = image_label_data["External ID"]
        image_path = os.path.join(
            yolov5_directories.training_directory, image_name)
        image = download_image(image_label_data["Labeled Data"])

        label_list: List[SegmentationData] = []

        for label_instance in image_label_data["Label"]["objects"]:
            class_name = label_instance["value"]
            raw_image_mask = download_image(label_instance["instanceURI"])
            gray_image_mask = cv2.cvtColor(raw_image_mask, cv2.COLOR_BGR2GRAY)

            segmentation_data = get_annotation(
                class_name_index[class_name], gray_image_mask)
            label_list.append(segmentation_data)
        write_yolo_image(image_path, image, label_list)
        print(f"Wrote Image {image_index} - {image_name} in "
              f"{time.time() - start_time} ")

    generate_yaml(yolov5_directories, class_names)


def generate_yolo(output_dir: str, json_path: str):
    """
    Generate and write data to Files following the YoloV5 data format
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
    https://github.com/ultralytics/yolov5/issues/2565

    Args:
        output_dir (str): directory that will contain the new folder with the
            newly formated data
        json_path (str): path to json file that contains the labelbox data
    """
    json_data = load_json(json_path)
    convert_format(json_data, output_dir)


if __name__ == "__main__":

    args = parse_args()

    generate_yolo(args.output_dir, args.box_label_file)
