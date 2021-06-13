import csv
import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import image_processing.load_images as load
import image_processing.processing as proc
import image_processing.stamina as stamina
import image_processing.globals as GV


import imutils
# Need this import to use imutils.contours.sort_contours,
#   without it Module raises AttributeError
from imutils import contours  # noqa
import statistics


def getLevelDigit(sourceImage, digitDict: np.array, train: bool = False):

    folders = ["L", "v", "l"]
    if train:
        trainLevels(image)

    results = {}
    for folder in folders:
        folderPath = os.path.join(GV.lvlPath, folder)
        folderFiles = os.listdir(folderPath)
        if folder not in results:
            results[folder] = []
        for file in folderFiles:
            filePath = os.path.join(folderPath, file)
            templateImage = cv2.imread(filePath)

            height, width = templateImage.shape[:2]

            x, y, w, h = cv2.boundingRect(digitDict["cnt"])
            ROI = sourceImage[y:y+h, x:x+w]

            heightRatio = h/height

            newHeight = int(height * heightRatio)
            newWidth = int(width * heightRatio)

            templateResize = cv2.resize(
                templateImage, (min(newWidth, w), min(newHeight, h)))

            templateMatch = cv2.matchTemplate(
                ROI, templateResize, cv2.TM_CCOEFF_NORMED)
            (_, score, _, _) = cv2.minMaxLoc(templateMatch)
            results[folder].append((score, filePath, height))
    returnResults = []
    for k, v in results.items():
        returnResults.append((max(v, key=lambda x: x[0]), k))
    return returnResults


def trainLevels(image):
    stamina.digitFeatures(image, saveDir=GV.lvlPath)


if __name__ == '__main__':

    csvfile = open(
        "/home/nate/projects/afk-image-processing/image_processing/si/si_data.txt",
        "r")

    fieldNames = ["path", "left", "bottom", "right", "top", "label"]
    reader = csv.DictReader(csvfile, fieldNames)
    id = 0
    data_list = []
    annotation_list = []
    print(reader)
    size_dict = {}
    count = 0

    hero_si_data = {}

    lvl_ratio_dict = {}

    for row in reader:
        label = row["label"]
        path = row["path"]
        name = os.path.basename(path)
        hero_si_data[name] = row
        # if label not in size_dict:
        #     size_dict[label] = {}
        #     size_dict[label]["width"] = 0
        #     size_dict[label]["height"] = 0
        #     size_dict[label]["count"] = 0

        image = cv2.imread(path)
        if image is None:
            continue
        hero_si_data[name]["image"] = image

        height, width = image.shape[:2]

        cropped = image[0:int(height*0.3), int(width*0.25):width]

        # (hMin = 11 , sMin = 4, vMin = 110), (hMax = 38 , sMax = 191, vMax = 255)
        lowerBound = np.array([11, 4, 167])
        upperBound = np.array([38, 191, 255])

        cropped_blur = proc.blur_image(
            cropped, hsv_range=[lowerBound, upperBound])
        hero_si_data[name]["cropped"] = cropped

        cnts = cv2.findContours(
            cropped_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cv2.findContours(
            cropped_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = imutils.contours.sort_contours(cnts,
                                              method="left-to-right")[0]
        # print(name)
        bins = {}
        all_y = []

        for index, digit in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(digit)
            if w > 3 and h > 10:
                bins[index] = {}
                bins[index]["cnt"] = digit
                all_y.append((y, y+h, index))
                # print(y)
                # ROI = cropped[y:y+h, x:x+w]
                # load.display_image(ROI)
                # sizedROI = cv2.resize(ROI, (57, 88))
        # print(sorted(all_y, key=lambda x: x[0]))

        top_med = statistics.median([num[0] for num in all_y])
        bottom_med = statistics.median([num[1] for num in all_y])

        # print(start_med)
        # print(end_med)

        final_digits = []

        for digitTuple in all_y:
            digitTop = digitTuple[0]
            digitBottom = digitTuple[1]
            index = digitTuple[2]
            if abs(digitBottom-bottom_med) <= 3 and (digitTop >= (top_med - 3)):
                x, y, w, h = cv2.boundingRect(bins[index]["cnt"])
                ROI = cropped[y:y+h, x:x+w]
                bins[index]["roi"] = ROI
                # load.display_image(ROI)
                final_digits.append(digitTuple)
        # images = [bins[digitTuple[2]]["roi"] for digitTuple in final_digits]
        # load.display_image([cropped] + images, multiple=True)
        # print(final_digits)
        verified = []
        for digitTuple in final_digits[:3]:
            index = digitTuple[2]
            image = bins[index]["roi"]
            bins[index]["info"] = digitTuple
            results = getLevelDigit(cropped, bins[index])
            bestMatch = max(results, key=lambda x: x[0][0])
            if bestMatch[0][0] < 0.9:
                break
            verified.append(digitTuple)
            height = image.shape[1]
            templateHeight = bestMatch[0][2]
            bins[index]["scale"] = height/templateHeight
            bins[index]["match_info"] = bestMatch

            # print("Scale", bestMatch, templateHeight,
            #       height, bins[index]["scale"])

        # load.display_image(cropped)
        left = int(row["left"])
        right = int(row["right"])
        top = int(row["top"])
        bottom = int(row["bottom"])

        si_width = right - left
        si_height = bottom - top

        heights = []
        for digitTuple in verified:
            digitTop = digitTuple[0]
            digitBottom = digitTuple[1]
            index = digitTuple[2]
            height = bins[index]["match_info"][0][2]
            # print(bins[index]["match_info"])
            # load.display_image(bins[index]["roi"])

            heights.append(height)
            name = bins[index]["match_info"][1]
            if name not in lvl_ratio_dict:
                lvl_ratio_dict[name] = {}

            if label not in lvl_ratio_dict[name]:
                lvl_ratio_dict[name][label] = []
            tempDigitDict = {"si": {}}
            tempDigitDict["height"] = height
            tempDigitDict["si"]["height"] = si_height
            tempDigitDict["si"]["width"] = si_width

            tempDigitDict["path"] = bins[index]["match_info"]

            v_scale = si_height/height
            h_scale = si_width/height
            tempDigitDict["v_scale"] = v_scale
            tempDigitDict["h_scale"] = h_scale

            lvl_ratio_dict[name][label].append(tempDigitDict)
    for digitName, digitDict in lvl_ratio_dict.items():
        for si_label, tempDicts in digitDict.items():
            print("{} {}".format(digitName, si_label))
            print("h_scale")
            print("max", max(tempDicts, key=lambda x: x["h_scale"]))
            print("min", min(tempDicts, key=lambda x: x["h_scale"]))
            print("v_scale")
            print("max", max(tempDicts, key=lambda x: x["v_scale"]))
            print("min", min(tempDicts, key=lambda x: x["v_scale"]))
    data = pd.DataFrame(columns=["Format"])
    header = ["digitName", "si_name", "h_scale", "v_scale"]
    for digitName, digitDict in lvl_ratio_dict.items():
        for si_label, tempDicts in digitDict.items():
            print("{} {} average".format(digitName, si_label))
            h_avg = np.mean([scale["h_scale"]
                             for scale in tempDicts])
            print("h_scale", h_avg)
            v_avg = np.mean([scale["v_scale"]
                             for scale in tempDicts])
            print("v_scale", v_avg)
            imageData = "{},{},{},{}".format(
                digitName, si_label, h_avg, v_avg)
            data.loc[len(data)] = [imageData]
    data.to_csv('lvl_txt_si_scale.txt',
                header=None, index=None, sep=' ')
    # print(lvl_ratio_dict)

    # size_dict[label]["width"] = size_dict[label]["width"] + width
    # size_dict[label]["height"] = size_dict[label]["height"] + height
    # print(label, width, height)
    # print(size_dict[label]["width"])
    # print(size_dict[label]["height"])

    # size_dict[label]["count"] = size_dict[label]["count"] + 1
    # print("count", count)

    # for label, size in size_dict.items():

    #     print(label, "width", size["width"]/size["count"])
    #     print(label, "height", size["height"]/size["count"])
