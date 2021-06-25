import csv
import cv2
import matplotlib.pyplot as plt
import image_processing.globals as GV
import os

if __name__ == "__main__":

    csvfile = open(
        "/home/nate/projects/afk-image-processing/image_processing/fi/fi_data.txt",
        "r")

    write_file = open('output.csv', 'w')

    fieldNames = ["path", "left", "bottom", "right", "top", "label"]
    reader = csv.DictReader(csvfile, fieldNames)
    id = 0
    data_list = []
    annotation_list = []
    size_dict = {}
    count = 0

    folders = os.listdir(GV.fi_train_path)
    folders = [folder.lower() for folder in folders]

    for row in reader:
        image_path = row["path"]
        image = cv2.imread(image_path)

        plt.ion()
        plt.figure()
        plt.imshow(image)
        plt.show()
        fi_label = input(
            "Please enter fi type: ")
        plt.close()

        if fi_label in folders:
            label = fi_label
        else:
            label = "none"
        write_path = os.path.join(GV.fi_train_path,
                                  label, os.path.basename(image_path))
        row["label"] = label
        row["path"] = write_path
        # cv2.imwrite(write_path, image)
        wr = csv.writer(write_file, dialect='excel')
        wr.writerow(row.values())
