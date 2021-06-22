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


def getDigit(image):
    """
    Detect and return 'L', 'v', 'l' from image

    Args:
        image: np.array of rgb hero image

    Return:
        list of dictionaries with information about each  'L', 'v', 'l'
        detected
    """

    height, width = image.shape[:2]

    cropped = image[0:int(height*0.3), int(width*0.25):width]

    # (hMin = 11 , sMin = 4, vMin = 110), (hMax = 38 , sMax = 191, vMax = 255)
    lowerBound = np.array([11, 4, 167])
    upperBound = np.array([38, 191, 255])

    cropped_blur = proc.blur_image(
        cropped, hsv_range=[lowerBound, upperBound])

    cnts = cv2.findContours(
        cropped_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cv2.findContours(
        cropped_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = imutils.contours.sort_contours(cnts, method="left-to-right")[0]
    bins = {}
    all_y = []

    for index, digit in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(digit)
        if w > 3 and h > 10:
            bins[index] = {}
            bins[index]["cnt"] = digit
            all_y.append((y, y+h, index))

    top_med = statistics.median([num[0] for num in all_y])
    bottom_med = statistics.median([num[1] for num in all_y])

    final_digits = []

    for digitTuple in all_y:
        digitTop = digitTuple[0]
        digitBottom = digitTuple[1]
        index = digitTuple[2]
        if abs(digitBottom-bottom_med) <= 3 and (digitTop >= (top_med - 3)):
            x, y, w, h = cv2.boundingRect(bins[index]["cnt"])
            ROI = cropped[y:y+h, x:x+w]
            bins[index]["roi"] = ROI
            final_digits.append(digitTuple)
    verified = {}
    # iterate over first 3 "letters" detected
    for digitTuple in final_digits[:3]:
        index = digitTuple[2]
        image = bins[index]["roi"]
        bins[index]["digit_info"] = digitTuple
        results = getLevelDigit(cropped, bins[index])
        v = list(results.values())
        k = list(results.keys())
        bestLabel = k[v.index(max(v))]
        # bestLabel = max(results.items(), key=lambda x: x[0])
        bestMatch = results[bestLabel]
        if bestMatch[0] < 0.9:
            continue
        height = image.shape[1]
        templateHeight = bestMatch[2]
        bins[index]["scale"] = height/templateHeight
        bins[index]["match_info"] = bestMatch
        # bins[index]["digit_info"] = digitTuple
        verified[bestLabel] = bins[index]
    return verified


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
    returnResults = {}
    for k, v in results.items():
        returnResults[k] = max(v, key=lambda x: x[0])
    return returnResults


def trainLevels(image):
    stamina.digitFeatures(image, saveDir=GV.lvlPath)


if __name__ == '__main__':

    # csvfile = open(
    #     "/home/nate/projects/afk-image-processing/image_processing/scripts/lvl_txt_si_scale.txt",
    #     "r")

    # header = ["digitName", "si_name", "v_scale"]

    # reader = csv.DictReader(csvfile, header)
    # si_data_dict = {}
    # for row in reader:
    #     digitName = row["digitName"]
    #     si_name = row["si_name"]
    #     # h_scale = float(row["h_scale"])
    #     v_scale = float(row["v_scale"])
    #     if si_name not in si_data_dict:
    #         si_data_dict[si_name] = {}
    #     si_data_dict[si_name][digitName] = {}
    #     # si_data_dict[si_name][digitName]["h_scale"] = h_scale
    #     si_data_dict[si_name][digitName]["v_scale"] = v_scale

    si_file = open(
        "/home/nate/projects/afk-image-processing/image_processing/si/si_data.txt",
        "r")

    si_field_names = ["path", "left", "bottom", "right", "top", "label"]
    si_reader = csv.DictReader(si_file, si_field_names)
    id = 0
    data_list = []
    annotation_list = []
    size_dict = {}
    count = 0

    # hero_si_data = {}

    lvl_ratio_dict = {}

    for row in si_reader:
        si_label = row["label"]
        path = row["path"]
        rowName = os.path.basename(path)

        image = cv2.imread(path)
        if image is None:
            print("Failed to load image {}".format(path))
            continue

        left = int(row["left"])
        right = int(row["right"])
        top = int(row["top"])
        bottom = int(row["bottom"])

        si_width = right - left
        si_height = bottom - top

        heights = []
        newHeights = []
        actualHeights = []

        verified = getDigit(image)

        for digitName, digitDict in verified.items():
            digitTuple = digitDict["digit_info"]
            digitTop = digitTuple[0]
            digitBottom = digitTuple[1]
            # index = digitTuple[2]
            bins = digitDict
            height = bins["match_info"][2]

            heights.append(height)
            if digitName not in lvl_ratio_dict:
                lvl_ratio_dict[digitName] = {}

            if si_label not in lvl_ratio_dict[digitName]:
                lvl_ratio_dict[digitName][si_label] = []
            tempDigitDict = {"si": {}}
            tempDigitDict["height"] = height
            tempDigitDict["si"]["height"] = si_height
            tempDigitDict["si"]["width"] = si_width

            tempDigitDict["path"] = bins["match_info"]

            v_scale = si_height/height
            # h_scale = si_width/height
            tempDigitDict["v_scale"] = v_scale
            # tempDigitDict["h_scale"] = h_scale

            # digitHeight = digitBottom - digitTop
        #     temp_v_scale = si_data_dict[si_label][digitName]["v_scale"]
        #     newHeights.append((digitHeight*temp_v_scale, digitName))
        #     actualHeights.append((digitHeight, temp_v_scale))

            lvl_ratio_dict[digitName][si_label].append(tempDigitDict)
        # print("si: {} {}".format(si_label, actualHeights))
        # print("Prediction - h: {}".format(newHeights))
        # print("Actual - h: {}".format(si_height))

    fi_file = open(
        "/home/nate/projects/afk-image-processing/image_processing/fi/fi_data.txt",
        "r")

    fi_field_names = ["path", "left", "bottom", "right", "top", "label"]
    fi_reader = csv.DictReader(fi_file, fi_field_names)
    id = 0
    data_list = []
    annotation_list = []
    size_dict = {}
    count = 0

    for row in fi_reader:
        fi_label = row["label"]
        path = row["path"]
        rowName = os.path.basename(path)

        image = cv2.imread(path)
        if image is None:
            print("Failed to load image {}".format(path))
            continue

        left = int(row["left"])
        right = int(row["right"])
        top = int(row["top"])
        bottom = int(row["bottom"])

        fi_width = right - left
        fi_height = bottom - top

        heights = []
        newHeights = []
        actualHeights = []

        verified = getDigit(image)

        for digitName, digitDict in verified.items():
            digitTuple = digitDict["digit_info"]
            digitTop = digitTuple[0]
            digitBottom = digitTuple[1]
            # index = digitTuple[2]
            bins = digitDict
            height = bins["match_info"][2]

            heights.append(height)
            if digitName not in lvl_ratio_dict:
                lvl_ratio_dict[digitName] = {}

            if fi_label not in lvl_ratio_dict[digitName]:
                lvl_ratio_dict[digitName][fi_label] = []
            tempDigitDict = {"fi": {}}
            tempDigitDict["height"] = height
            tempDigitDict["fi"]["height"] = fi_height
            tempDigitDict["fi"]["width"] = fi_width

            tempDigitDict["path"] = bins["match_info"]

            v_scale = fi_height/height
            # h_scale = si_width/height
            tempDigitDict["v_scale"] = v_scale
            # tempDigitDict["h_scale"] = h_scale

            # digitHeight = digitBottom - digitTop
        #     temp_v_scale = si_data_dict[si_label][digitName]["v_scale"]
        #     newHeights.append((digitHeight*temp_v_scale, digitName))
        #     actualHeights.append((digitHeight, temp_v_scale))

            lvl_ratio_dict[digitName][fi_label].append(tempDigitDict)

    for digitName, digitDict in lvl_ratio_dict.items():
        for si_label, tempDicts in digitDict.items():
            print("{} {}".format(digitName, si_label))
            # print("h_scale")
            # print("max", max(tempDicts, key=lambda x: x["h_scale"]))
            # print("min", min(tempDicts, key=lambda x: x["h_scale"]))
            print("v_scale")
            print("max", max(tempDicts, key=lambda x: x["v_scale"]))
            print("min", min(tempDicts, key=lambda x: x["v_scale"]))
    data = pd.DataFrame(columns=["Format"])
    for digitName, digitDict in lvl_ratio_dict.items():
        for si_label, tempDicts in digitDict.items():
            print("{} {} average".format(digitName, si_label))
            # h_avg = np.mean([scale["h_scale"]
            #                  for scale in tempDicts])
            # print("h_scale", h_avg)
            v_avg = np.mean([scale["v_scale"]
                             for scale in tempDicts])
            print("v_scale", v_avg)
            imageData = "{},{},{}".format(
                digitName, si_label, v_avg)
            data.loc[len(data)] = [imageData]
    header = ["digitName", "si_name", "v_scale"]
    data.to_csv('lvl_txt_si_scale.txt',
                header=None, index=None, sep=' ')
