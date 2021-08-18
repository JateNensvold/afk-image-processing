import image_processing.globals as GV
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2

if __name__ == "__main__":
    path = GV.siPath
    trainPath = os.path.join(path, "train")
    data = pd.DataFrame(columns=["Format"])
    trainPathList = os.listdir(trainPath)
    trainPathList = sorted(trainPathList)
    for folder in trainPathList:
        folderPath = os.path.join(trainPath, folder)
        folderPathList = os.listdir(folderPath)
        folderPathList = sorted(folderPathList)
        for imageName in folderPathList:
            newInput = "n"

            while "n" in newInput:
                trainingImagePath = os.path.join(folderPath, imageName)
                trainingImage = cv2.imread(trainingImagePath)
                plt.ion()
                plt.figure()
                plt.imshow(trainingImage)
                plt.show()
                points = input(
                    "Please enter left, top, right, bottom points: ")
                plt.close()

                points = points.replace(r"/\s\s+/g", ' ')

                left, top, right, bottom = points.split(" ")[:4]
                ROI = trainingImage[int(top):int(int(bottom)),
                                    int(left):int(int(right))]
                plt.ion()
                plt.figure()
                plt.imshow(ROI)
                plt.show()

                newInput = input("Enter(n) to redo bounds, press any other"
                                 " key to continue: ")
                plt.close()

            imageData = "{},{},{},{},{},{}".format(
                trainingImagePath, left, bottom, right, top, folder)
            data.loc[len(data)] = [imageData]
            data.to_csv('si_data.txt', header=None, index=None, sep=' ')
