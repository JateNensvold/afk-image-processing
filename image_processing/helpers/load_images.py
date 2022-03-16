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
        numpy.ndarray of rgb elements
    """

    lower_image_path = image_path.lower()
    if "discordapp" in lower_image_path and lower_image_path.endswith(
            (".png", ".jpg")):

        resp = requests.get(image_path, stream=True).raw
        raw_image = np.asarray(bytearray(resp.read()), dtype="uint8")

        return cv2.imdecode(raw_image, cv2.IMREAD_COLOR)

    elif check_path is False or os.path.exists(image_path):
        return cv2.imread(image_path)
    else:
        raise FileNotFoundError(image_path)
