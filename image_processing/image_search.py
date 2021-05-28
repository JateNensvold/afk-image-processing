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
    files = load.findFiles("./yuna/*", flag=True)
    print("Files found", files)
    images = {}
    difference = {}
    names = {}
    image = cv2.imread(img2_path)
    import image_processing.processing as processing
    import image_processing.load_images as classify
    heroes = processing.getHeroes(image)
    for name, hero_image in heroes.items():
        plt.imshow(hero_image[0])
        plt.show()
        alpha, mask = classify.colorClassify(hero_image[0], hero_image[1])
        cv2.imwrite("./temp/" + name, hero_image[0])

    for name in files:
        Image.save("./temp/" + name)
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
    print("Closest hero distance:", sorted_hash[0])
    print("Closest hero: ", difference[sorted_hash[0]])
    print("Lorsan hamming dist:", names["./yuna/lorsan.png"])


if __name__ == "__main__":
    # import glob
    # output = glob.glob("../lorsan*")
    # image = cv2.imread("../test_ss.png")

    image_search("../lorsan.png", "../test_ss.png")
