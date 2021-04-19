import numpy as np
import cv2
from matplotlib import pyplot as plt

import image_processing.load_images as load

from PIL import Image
import imagehash


def build_database():
    pass


def image_search(img1_path: str, img2_path: str):
    print("Search image:", img1_path)
    hash1 = imagehash.average_hash(Image.open(img1_path))
    files = load.findFiles("./yuna/lorsan*", flag=True)
    print("Files found", files)
    images = {}
    difference = {}
    names = {}
    for name in files:
        img = Image.open(name)
        hash = imagehash.average_hash(img)
        images[hash] = {}
        images[hash]["name"] = name
        images[hash]["img"] = img
        hashdiff = hash1-hash
        if hashdiff not in difference:
            difference[hashdiff] = []
        difference[hashdiff].append(name)
        names[name] = hashdiff
    sorted_hash = sorted(difference.keys())
    print(sorted_hash)
    print(sorted_hash[0])
    print(difference[sorted_hash[0]])
    print("Lorsan hamming dist:", names["../yuna/lorsan.png"])


if __name__ == "__main__":
    # import glob
    # output = glob.glob("../lorsan*")
    image_search("../lorsan.png", "../heroes/lorsan.jpg")
