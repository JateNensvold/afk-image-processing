import os
import requests

import cv2
import numpy as np


def load_image(image_path: str, check_path=True) -> np.ndarray:
    """
    Loads image from image path
    Args:
        image_path: path to image on disk
        check_path: flag to raise an error if 'image_path' is not found
    Returns:
        numpy.ndarray of BGR elements
    """

    lower_image_path = image_path.lower()
    if "discordapp" in lower_image_path and lower_image_path.endswith(
            (".png", ".jpg", ".webp")):

        resp = requests.get(image_path, stream=True).raw
        raw_image = np.asarray(bytearray(resp.read()), dtype="uint8")

        output = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
        return output

    elif check_path is False or os.path.exists(image_path):
        output = cv2.imread(image_path)
        return output
    else:
        raise FileNotFoundError(image_path)
