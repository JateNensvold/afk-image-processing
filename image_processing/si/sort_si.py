import os
import image_processing.processing as pr
import image_processing.globals as GV
import image_processing.build_db as BD
import numpy as np
import cv2
import matplotlib.pyplot as plt


def sortTrainHeroImage(name: str, image: np.array):
    siPath = GV.siPath
    trainPath = os.path.join(siPath, "train", )

    print("Please enter the SI of the hero show: ")

    plt.imshow(image)
    plt.show()
    siFolder = input()

    trainFolder = os.path.join(trainPath, siFolder, name)
    cv2.imwrite(trainFolder, image)


if __name__ == "__main__":

    imageDB = BD.buildDB()
    siPath = GV.siPath

    siTempPath = os.path.join(siPath, "temp")

    for imagePath in os.listdir(siTempPath):
        rosterImage = cv2.imread(os.path.join(GV.siPath, "temp", imagePath))
        heroes = pr.getHeroes(rosterImage)
        # cropHeroes = load.crop_heroes(heroes)

        for name, imageDict in heroes.items():
            heroImage = imageDict["image"]

            heroLabel, _ = imageDB.search(heroImage)
            sortTrainHeroImage("{}{}".format(heroLabel, name), heroImage)
