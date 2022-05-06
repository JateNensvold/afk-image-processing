
import traceback
from typing import List, Callable

import cv2

import numpy as np
from numpy import asarray

from image_processing.utils.color_helper import MplColorHelper, RGBColor
from image_processing.load_images import display_image
from image_processing.database.config import Config


class LabelMixin:
    """_summary_
    """

    def __init__(self, config: Config, labels, save_callback: Callable):
        """_summary_

        Args:
            config (_type_): _description_
        """
        self.config = config
        self.labels = labels
        self.save = save_callback

    def label_add(self, ascension_data: RGBColor,
                  display_images: List[np.ndarray] = None):
        """_summary_

        Args:
            ascension_data (AscensionData): _description_
        """
        color_map = MplColorHelper()
        label_palette = np.zeros(shape=[400, 400, 3], dtype=np.uint8)
        label_palette[0:400, :, :] += np.uint(asarray(ascension_data))

        cv2.rectangle(label_palette, (1, 1), (label_palette.shape[:2]),
                      [*color_map.get_rgb("red")], 10)

        if display_images is None:
            display_images = []

        display_images.append(label_palette)

        update = False
        while not update:
            print(f"Data: {ascension_data}")

            print("Label the above data with one of the following options, "
                  "or enter 's' to skip:")
            for label_index, label in enumerate(self.labels, start=1):
                print(f"{label_index}: {label}")
            new_label = display_image(
                display_images, display=True, message="Index: ")
            try:
                if new_label.lower()[0] == "s":
                    return
                label_index = int(new_label) - 1
                if 0 <= label_index < (len(self.labels)):
                    self._update_database(
                        self.labels[label_index], ascension_data)
                    return self.labels[label_index]
            except Exception as _exception:  # pylint: disable=broad-except
                print(traceback.format_exc())
                print("Please enter a valid index for the labels "
                      f"provided({self.labels}), ({new_label}) is not a"
                      "valid index")

    def _update_database(self, label: str, ascension_data: RGBColor):
        """_summary_

        Args:
            label (str): _description_
            ascension_data (AscensionData): _description_
        """
        self.config[label].add(ascension_data)
        self.save()
