import os
import json
import cv2

import numpy as np
import matplotlib.pyplot as plt
import image_processing.load_images as load
import image_processing.processing as processing
import image_processing.globals as GV
import imutils
# Need this import to use imutils.contours.sort_contours,
#   without it Module raises AttributeError
from imutils import contours  # noqa


def sort_row(row: list, heroes: dict):
    return sorted(row, key=lambda x: heroes[x[1]]["dimensions"]["x"][0])


def generate_rows(heroes: dict, spacing: int = 10):
    rows = []
    heads = {}

    for k, v in heroes.items():
        y = v["dimensions"]["y"][0]

        closeRow = False
        index = None
        for head, headIndex in heads.items():
            if abs(head - y) < spacing:
                closeRow = True
                index = headIndex
                break

        if not closeRow:
            rows.append([])
            heads[y] = len(rows) - 1
            index = heads[y]
        rows[index].append((v["label"], k))
    rows = sorted(rows, key=lambda x: heroes[x[0][1]]["dimensions"]["y"][0])
    for i in range(len(rows)):
        newRow = sort_row(rows[i], heroes)
        rows[i] = newRow

    return rows


def get_stamina_area(rows: list, heroes: dict, sourceImage: np.array):
    numRows = len(rows)
    staminaCount = {}
    # iterate across length of row
    averageHeight = 0
    samples = 0
    for j in range(numRows):
        for i in range(len(rows[j])):
            # iterate over column
            # Last row
            unitName = rows[j][i][1]
            y = heroes[unitName]["dimensions"]["y"]
            x = heroes[unitName]["dimensions"]["x"]
            gapStartX = x[0]
            gapStartY = y[1]
            gapWidth = x[1] - x[0]
            if (j + 1) == numRows:
                gapBottom = gapStartY + int(averageHeight)
            else:
                gapBottom = heroes[rows[j+1][i][1]]["dimensions"]["y"][0]

                samples += 1
                a = 1/samples
                b = 1 - a
                averageHeight = (a * (gapBottom - gapStartY)
                                 ) + (b * averageHeight)
            staminaArea = sourceImage[gapStartY:gapBottom,
                                      gapStartX:gapStartX + gapWidth]
            staminaCount[unitName] = staminaArea

    return staminaCount


def get_text(staminaAreas: dict, train: bool = False):

    # build template dictionary
    digits = {}
    numbersFolder = GV.numbersPath
    # numbersFolder = os.path.join(os.path.dirname(
    #     os.path.abspath(__file__)), "numbers")

    referenceFolders = os.listdir(numbersFolder)
    for folder in referenceFolders:
        if folder not in digits:
            digits[folder] = {}
        digitFolder = os.path.join(numbersFolder, folder)
        for i in os.listdir(digitFolder):
            name, ext = os.path.splitext(i)
            digits[folder][name] = cv2.imread(os.path.join(digitFolder, i))
    output = {}
    for name, stamina_image in staminaAreas.items():
        original = stamina_image.copy()

        lower = np.array([0, 0, 176])
        upper = np.array([174, 34, 255])
        hsv = cv2.cvtColor(stamina_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(original, original, mask=mask)
        print("original", original.shape)
        print("mask", mask.shape)
        result[mask == 0] = (255, 255, 255)

        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.threshold(
            result, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

        digitCnts = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = imutils.grab_contours(digitCnts)
        digitCnts = imutils.contours.sort_contours(digitCnts,
                                                   method="left-to-right")[0]
        digitText = []
        for digit in digitCnts:
            x, y, w, h = cv2.boundingRect(digit)
            if w > 6 and h > 12:
                ROI = stamina_image[y:y+h, x:x+w]
                sizedROI = cv2.resize(ROI, (57, 88))
                if train:
                    digitFeatures(sizedROI)
                else:

                    numberScore = []

                    for digitName, digitDICT in digits.items():
                        scores = []
                        for digitIteration, digitImage in digitDICT.items():
                            templateMatch = cv2.matchTemplate(
                                sizedROI, digitImage, cv2.TM_CCOEFF)
                            (_, score, _, _) = cv2.minMaxLoc(templateMatch)
                            scores.append(score)
                        avgScore = sum(scores)/len(scores)
                        numberScore.append((digitName, avgScore))
                    temp = sorted(
                        numberScore, key=lambda x: x[1], reverse=True)
                    digitText.append(temp[0][0])

        text = "".join(digitText)
        output[name] = text
    return output


def signatureItemFeatures(hero: np.array):
    x, y, _ = hero.shape
    print(x, y)
    x_div = 2.8
    y_div = 2.0
    offset = 5
    newHero = hero[offset: int(y/y_div), offset: int(x/x_div)]

    print("hero", hero.shape)

    si_dict = {}
    baseSIDir = GV.siPath

    siFolders = os.listdir(os.path.join(baseSIDir, "base"))

    imgCopy1 = newHero.copy()
    grayCopy = cv2.cvtColor(imgCopy1, cv2.COLOR_BGR2GRAY)

    for folder in siFolders:
        SIDir = os.path.join(baseSIDir, folder)
        SIPhotos = os.listdir(SIDir)
        for imageName in SIPhotos:

            siImage = cv2.imread(os.path.join(baseSIDir, folder, imageName))
            if folder not in si_dict:
                si_dict[folder] = {}
            # si_dict[folder][imageName] = siImage

            SIGray = cv2.cvtColor(siImage, cv2.COLOR_BGR2GRAY)

            si_dict[folder]["image"] = siImage
            circles = cv2.HoughCircles(SIGray,
                                       cv2.HOUGH_GRADIENT,
                                       # resolution of accumulator array.
                                       dp=2.5,
                                       minDist=100,
                                       # number of pixels center of circles should be from each other, hardcode
                                       param1=50,
                                       param2=100,
                                       # HoughCircles will look for circles at minimum this size
                                       minRadius=(70),
                                       # HoughCircles will look for circles at maximum this size
                                       maxRadius=(83))

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(siImage, (i[0], i[1]), i[2], (0, 255, 0), 2)

                    si_dict[folder]["circle"] = i
                    si_dict[folder]["center"] = i[:2]

    circles = cv2.HoughCircles(grayCopy,
                               cv2.HOUGH_GRADIENT,
                               # resolution of accumulator array.
                               dp=1.0,
                               minDist=100,  # number of pixels center of circles should be from each other, hardcode
                               param1=50,
                               param2=34,
                               # HoughCircles will look for circles at minimum this size
                               minRadius=(25),
                               # HoughCircles will look for circles at maximum this size
                               maxRadius=(38))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(hero, (i[0]+offset, i[1]+offset), i[2], (0, 255, 0), 2)
            heroCircle = i

    plt.imshow(hero)
    plt.show()

    lower = np.array([0, 25, 00])
    upper = np.array([73, 255, 255])
    # (hMin = 0 , sMin = 18, vMin = 88), (hMax = 28 , sMax = 182, vMax = 255)
    hsv = cv2.cvtColor(newHero, cv2.COLOR_BGR2HSV)
    # plt.imshow(hsv)
    # plt.show()
    mask = cv2.inRange(hsv, lower, upper)
    # plt.imshow(mask)
    # plt.show()
    print("mask", mask.shape)
    result = cv2.bitwise_and(newHero, newHero, mask=mask)
    result[mask == 0] = (255, 255, 255)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.threshold(
        result, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    # plt.imshow(mask)
    # plt.show()
    digitCnts = cv2.findContours(
        result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = imutils.grab_contours(digitCnts)
    digitCnts = imutils.contours.sort_contours(digitCnts,
                                               method="left-to-right")[0]

    numberScore = []
    # for digit in digitCnts:
    digit = digitCnts[0]

    x, y, w, h = cv2.boundingRect(digit)
    ROI = newHero[y:y+h, x:x+w]

    print(ROI.shape)

    # if w > 30 and h > 30:

    for folderName, imageDict in si_dict.items():
        scores = []
        # for siName, siImage in imageDict.items():
        siImage = imageDict["image"]
        center = imageDict["center"]
        circle = imageDict["circle"]
        si_W, si_H, _ = siImage.shape

        center_x = center[0]
        center_y = center[1]

        radius = circle[2]

        circleRatio = heroCircle[2]/radius

        # print("{} left: {} right: {}".format(
        #     folderName,
        #     center_x/si_W, (si_W-center_x) / si_W))
        # print("top: {} bottom: {}".format(
        #     center_y/si_H, (si_H - center_y)/si_H))

        si_ratio = si_H / si_W
        x, _, _ = newHero.shape
        finalHero = hero[offset: int(
            offset + x), offset: int(offset + x*si_ratio)]
        plt.imshow(finalHero)
        plt.show()

        x, y, _ = newHero.shape
        sizedROI = cv2.resize(
            newHero, (int(x * circleRatio), int(y * circleRatio)))
        # print(sizedROI.shape, siImage.shape)
        templateMatch = cv2.matchTemplate(
            sizedROI, siImage, cv2.TM_CCOEFF_NORMED)
        print("{}".format(folderName), templateMatch)
        (_, score, _, _) = cv2.minMaxLoc(templateMatch)
        scores.append(score)

        avgScore = sum(scores)/len(scores)
        numberScore.append((folderName, avgScore))
    temp = sorted(
        numberScore, key=lambda x: x[1], reverse=True)
    print("Best SI: {}".format(temp[0][0]))
    print("Others: {}".format(temp))
    firstKey = list(si_dict[temp[0][0]].keys())
    img_list = []

    bestMatch = si_dict[temp[0][0]][firstKey[0]]
    img_list.append(bestMatch)
    img_list.append(finalHero)
    img_list.append(sizedROI)
    cat_image = load.concat_resize(img_list)
    # plt.imshow(cat_image)
# def furnitureItemFeatures(hero: np.array):


def digitFeatures(digit: np.array):
    """
    Save a presized digit to whatever number is entered
    Args:
        digit: presized image of a digit, that will be saved as a training
            template under whatever digitName/label is entered when prompted
            by terminal
    Return:
        None
    """

    baseDir = GV.numbersPath
    os.listdir(baseDir)
    plt.figure()
    plt.imshow(digit)
    print("Please enter the number shown in the image after you close it: ")
    plt.show()
    number = input()
    numberDir = os.path.join(baseDir, number)
    numberLen = str(len(os.listdir(numberDir)))
    numberName = os.path.join(numberDir, numberLen)

    cv2.imwrite("{}.png".format(numberName), digit)


if __name__ == "__main__":

    # Load in base truth/reference images
    files = load.findFiles("../hero_icon/*")
    baseImages = []
    for i in files:
        hero = cv2.imread(i)
        baseName = os.path.basename(i)
        name, ext = os.path.splitext(baseName)

        baseImages.append((name, hero))

    # load in screenshot of heroes
    stamina_image = cv2.imread("./stamina.jpg")
    heroesDict = processing.getHeroes(stamina_image)

    cropHeroes = load.crop_heroes(heroesDict)
    imageDB = load.build_flann(baseImages)

    for k, v in cropHeroes.items():

        name, baseHeroImage = imageDB.search(v, display=False)
        heroesDict[k]["label"] = name

    rows = generate_rows(heroesDict)
    staminaAreas = get_stamina_area(rows, heroesDict, stamina_image)
    staminaOutput = get_text(staminaAreas)
    output = {}

    for name, text in staminaOutput.items():
        label = heroesDict[name]["label"]
        if label not in output:
            output[label] = {}
        output[label]["stamina"] = text

    outputJson = json.dumps(output)
    print(outputJson)
