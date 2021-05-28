import image_processing.globals as GV
import os

if __name__ == "__main__":
    path = GV.siPath
    trainPath = os.path.join(path, "train")

    for folder in os.listdir(trainPath):
        folderPath = os.path.join(trainPath, folder)
        for trainingImagePath in os.listdir(folderPath):
            