import cv2
import os

import numpy as np


def load_image(image_path: str, check_path=True) -> np.ndarray:
    """
    Loads image from image path
    Args:
        image_path: path to image on disk
        check_path: flag to raise an error if 'image_path' is not found
    Returns:
        numpy.ndarray of rgb elements
    """
    if check_path is False or os.path.exists(image_path):
        return cv2.imread(image_path)
    else:
        raise FileNotFoundError(image_path)
