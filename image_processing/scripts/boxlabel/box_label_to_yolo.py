import itertools
import os
import pathlib
import time
from typing import Dict, Iterator, List, NamedTuple
from multiprocessing.dummy import Pool
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

from image_processing.scripts.boxlabel.utils.input_arguments import (
    BOX_LABEL_DATA_FORMAT, BOX_LABEL_INSTANCE, parse_args, load_json)
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
        yaml_file.write(f"nc: {len(class_names)}\n")
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

    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    file_name, _extension_name = os.path.splitext(image_path)
    width, height = image.shape[:2]
    text_file_name = f"{file_name}.txt"
    with open(text_file_name, "w", encoding="utf-8") as segmentation_file:
        for segmentation_data in segmentation_data_list:
            segment_point_list: List[str] = []
            for segment_index, segment in enumerate(segmentation_data.segmentation):
                if segment_index % 2 == 0:
                    segment_width_location = segment/width
                    segment_point_list.append(str(segment_width_location))
                else:
                    segment_height_location = segment/height
                    segment_point_list.append(str(segment_height_location))

            annotation_str = ' '.join(segment_point_list)
            segmentation_file.write(
                f"{segmentation_data.segmentation_class_index} "
                f"{annotation_str} \n")


def process_yolo_image(image_index: int,
                       image_label_data: BOX_LABEL_INSTANCE,
                       image_directory: str,
                       class_name_index: Dict[str, int]):
    """_summary_

    Args:
        image_directory (str): _description_
        class_name_index (Dict[str, int]): _description_
        image_index (int): _description_
        image_label_data (BOX_LABEL_INSTANCE): _description_
    """
    start_time = time.time()
    image_name = image_label_data["External ID"]
    image_path = os.path.join(
        image_directory, image_name)
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
    print(f"Wrote Image {image_index} - {image_name[:20]} to "
          f"{image_directory[-30:]} in {time.time() - start_time} ")


def convert_format(json_data: BOX_LABEL_DATA_FORMAT, parent_dir: str,
                   train_validate_split: int = 0.9):
    """
    Convert data from labelbox format to YoloV5 format

    Args:
        json_data (BOX_LABEL_DATA_FORMAT): labelbox json data
        parent_dir (str): directory that will contain the new folder with the
            newly formated data
        train_validate_split (int): the frequency to split each class in
            json_data into training and validation data
    """
    start_time = time.time()
    output_dir = os.path.abspath(os.path.join(parent_dir, "yolo_data"))

    yolov5_directories = YoloV5Files(output_dir)
    yolov5_directories.create_directories()

    class_name_index: Dict[str, int] = {}
    class_data = get_classes(json_data)

    for class_index, class_name in enumerate(class_data.class_names):
        class_name_index[class_name] = class_index

    available_threads = (len(os.sched_getaffinity(0)) - 1)
    pool: ThreadPool = Pool(available_threads)

    class_instance_count = 0
    for class_name, class_instances in class_data.class_distribution.items():
        train_count = int(train_validate_split * len(class_instances))

        print(f"Class name: {class_name} Count: {len(class_instances)} "
              f"Distribution: {train_count}/{len(class_instances)-train_count}")

        training_instances, validation_instances, _test_instances = np.split(
            class_instances, [train_count, len(class_instances)])

        function_arguments_training = zip(
            list(range(class_instance_count,
                 class_instance_count + len(training_instances))),
            training_instances,
            itertools.repeat(yolov5_directories.training_directory),
            itertools.repeat(class_name_index))

        class_instance_count += len(training_instances)
        function_arguments_validation = zip(
            list(range(class_instance_count,
                 class_instance_count + len(validation_instances))),
            validation_instances,
            itertools.repeat(yolov5_directories.validation_directory),
            itertools.repeat(class_name_index))
        class_instance_count += len(validation_instances)

        pool.starmap(process_yolo_image, function_arguments_training)
        pool.starmap(process_yolo_image, function_arguments_validation)

    generate_yaml(yolov5_directories, class_data.class_names)
    print(f"Total Time: {time.time() - start_time}")


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
