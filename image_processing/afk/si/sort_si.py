import os
import image_processing.globals as GV
import numpy as np
import cv2
import matplotlib.pyplot as plt


def sortTrainHeroImage(name: str, image: np.array):
    trainPath = os.path.join(GV.si_path, "train", )

    print("Please enter the SI of the hero show: ")

    plt.imshow(image)
    plt.show()
    siFolder = input()

    trainFolder = os.path.join(trainPath, siFolder, name)
    cv2.imwrite(trainFolder, image)
