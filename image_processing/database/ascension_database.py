import collections
from typing import Dict
import cv2

import numpy as np
from faiss import index_factory
from faiss.swigfaiss import IndexIDMap

import image_processing.globals as GV
from image_processing.models.model_attributes import ModelResult
from image_processing.database.config import Config

from image_processing.utils.types.list_utils import UtilList
from image_processing.utils.color_helper import MplColorHelper, RGBLabel, RGBColor
from image_processing.database.mixin.label_mixin import LabelMixin
from image_processing.database.mixin.contour_mixin import ContourMixin

from .configs.ascension_constants import (
    ALL_ASCENSION_HSV_RANGE, ASCENSION_COLOR_DIMENSIONS, ASCENSION_TYPES)
INDEX_KEY = "IDMap,Flat"

TEXT_COLOR = MplColorHelper().get_rgb("red")


class AscensionConfig(Config):
    """_summary_

    Args:
        Config (_type_): _description_
    """
    class_name = "ascension"

    def __init__(self, name: str):
        """
        Wrapper around default Json Config, that generates the loads/generates
            the required keys to make a valid ascension_values.json file

        Args:
            name (str): _description_
        """
        super().__init__(name)
        if self.class_name not in self._db:
            self._db[self.class_name] = {}
        class_dict = self._db[self.class_name]
        for ascension_type in ASCENSION_TYPES:
            if ascension_type not in class_dict:
                class_dict[ascension_type] = set()
        self.save()


class AscensionSearch(LabelMixin, ContourMixin):
    """_summary_
    """

    def __init__(self, ascension_config: Config = None):
        """_summary_
        """

        self.index: IndexIDMap = index_factory(
            ASCENSION_COLOR_DIMENSIONS, INDEX_KEY)
        self.index_lookup: Dict[int, str] = {}

        self.labels_count = 0
        self.ascension_config = ascension_config
        labels = set()

        for ascension_name, ascension_data_set in ascension_config["ascension"].items():
            labels.add(ascension_name)
            for ascension_data in ascension_data_set:

                self.add_result(RGBLabel(ascension_name, *ascension_data))
        self.labels = list(labels)

        super().__init__(
            self.ascension_config[self.ascension_config.class_name],
            self.labels,
            self.ascension_config.save)

    def add_result(self, ascension_data: RGBLabel):
        """_summary_

        Args:
            ascension_label (RGBLabel): _description_

        Returns:
            _type_: _description_
        """
        self.labels_count += 1
        self.index_lookup[self.labels_count] = ascension_data.name
        self.index.add_with_ids(
            np.asarray([ascension_data.data()], dtype=np.float32),
            np.array([self.labels_count], np.int64))

    @classmethod
    def from_json(cls, ascension_json_path: str):
        """_summary_

        Args:
            json_data (Dict[str]): _description_
        """
        ascension_data = AscensionConfig(ascension_json_path)
        return AscensionSearch(ascension_data)

    def search_image(self, search_image: np.ndarray, **kwargs):
        """_summary_

        Args:
            search_image (np.ndarray): _description_
        """

        hsv_result_dict = self.create_contour_dict(
            search_image, ALL_ASCENSION_HSV_RANGE)
        ascension_result_counter = collections.Counter()

        for _contour_size, contour_list in hsv_result_dict.items():
            centroids, ascension_mask = self.draw_contour(
                search_image, contour_list)

            # create an image to help visualize the colors associated with
            #   each centroid
            display_image_list = None
            if "manual_update" in kwargs:

                dom_pallette = np.zeros(
                    shape=search_image.shape, dtype=np.uint8)
                last_index = 0
                for index, centroid in enumerate(centroids):
                    color_index = int(
                        (index+1) * search_image.shape[0]/len(centroids))
                    dom_pallette[last_index:color_index,
                                 :, :] += np.uint8(centroid)
                    cv2.rectangle(dom_pallette,
                                  [0, last_index],
                                  [search_image.shape[0], color_index],
                                  [*TEXT_COLOR],
                                  10)
                    last_index = color_index
                display_image_list = [
                    cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB),
                    ascension_mask,
                    dom_pallette]
            kwargs["display_image_list"] = display_image_list
            for index, centroid in enumerate(centroids):
                ascension_results = self.search(
                    RGBColor(*centroid),
                    **kwargs)
                ascension_result_counter.update(
                    [ascension_result.label
                        for ascension_result in ascension_results])
        print(ascension_result_counter)
        return ascension_result_counter

    def search(self,
               ascension_data: RGBColor,
               k: int = 10,
               distance_threshold=2000,
               manual_update: bool = True,
               auto_label: bool = True,
               display_image_list: list = None):
        """_summary_

        Args:
            ascension_data (AscensionData): object representing the HSV color
                of an Ascension Border
            k (int): number of search results to return (capped by length of
                search database)
            manual_update (bool): flag to allow for manually updating/adding
                a label to the ascensino database for the given `ascension_data` 
            auto_label (bool): flag to auto append labels to AscensionDatabase
                when the nearest k search results are the same and manual_update
                    is false
        """

        distance_result_array, index_result_array = self.index.search(
            np.asarray([ascension_data], dtype=np.float32),
            k=k)

        distance_result_array = distance_result_array[0]
        index_result_array = index_result_array[0]

        model_result_list: UtilList[ModelResult] = UtilList()

        for result_index, distance_result in enumerate((distance_result_array)):
            index_result = index_result_array[result_index]

            # Invalid index, not enough matches to filfill k
            if index_result == -1:
                break

            # Remaining results are too far away from distance threshhold
            if distance_result > distance_threshold:
                break

            result_label = self.index_lookup[index_result]
            # Rename result to only the first letter of its name
            # result_label = raw_result_label[0].upper()

            model_result_list.append(
                ModelResult(result_label, distance_result))

        print(model_result_list)

        counter = collections.Counter(
            [model_result.label for model_result in model_result_list])
        most_common_result_list = counter.most_common(1)

        if len(most_common_result_list) == 0:
            if manual_update:
                result_label = None
                result_count = 0
                return model_result_list
            else:
                if GV.verbosity(1):
                    print("Unable to find ascension match "
                          f"{len(most_common_result_list)}/{k}")
                return model_result_list
        else:
            result_label, result_count = most_common_result_list[0]

        if GV.verbosity(1):
            print(f"{result_label} count: {result_count}"
                  f"/{len(model_result_list)}/{k} (label count/result count/k)")

        # Update label in Ascension_database
        if manual_update:
            ascension_label = self.label_add(
                ascension_data, display_image_list)
            if ascension_label is not None:
                self.add_result(RGBLabel(ascension_label, *ascension_data))
        elif auto_label and result_count == k:
            if GV.verbosity(1):
                print(f"Auto-Adding label {result_label}")
            ascension_label = RGBLabel(result_label, *ascension_data)
            self.add_result(ascension_label)

        return model_result_list
