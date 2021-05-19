import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import image_processing.load_images as load

import image_processing.processing as processing
import pytesseract
import imutils

# pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'


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
    print(rows)

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
                # gapBottom = sourceImage.shape[1]
                gapBottom = gapStartY + int(averageHeight)
            else:
                print(heroes.keys())
                gapBottom = heroes[rows[j+1][i][1]]["dimensions"]["y"][0]

                samples += 1
                a = 1/samples
                b = 1 - a
                averageHeight = (a * (gapBottom - gapStartY)
                                 ) + (b * averageHeight)
            staminaArea = sourceImage[gapStartY:gapBottom,
                                      gapStartX:gapStartX + gapWidth]
            staminaCount[unitName] = staminaArea

            # cv2.rectangle(sourceImage, (gapStartX, gapStartY),
            #               (gapStartX + gapWidth, gapBottom),
            #               (0, 0, 255), 3)

            # print(staminaArea.shape)
            # plt.figure()
            # plt.imshow(sourceImage)
            # plt.show()

            # plt.figure()
            # plt.imshow(staminaArea)
            # plt.show()

        # print("({},{})x({},{})".format(x_30, x-x_30, y_30, y-y_30))
        # crop_img = output[y_30: y-y_30, x_30: x-x_30]
    return staminaCount


def get_text(staminaAreas: dict):
    for name, stamina_image in staminaAreas.items():
        original = stamina_image.copy()
# (hMin = 0 , sMin = 0, vMin = 176), (hMax = 174 , sMax = 34, vMax = 255)
        lower = np.array([0, 0, 176])
        upper = np.array([174, 34, 255])
        hsv = cv2.cvtColor(stamina_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask == 0] = (255, 255, 255)

        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.threshold(
            result, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

        # text = pytesseract.image_to_string(
        #     result, lang='eng', config='--psm 11')
        # print("name: {} stamina: {}".format(name, text))
        # loop over the OCR-A reference contours

        contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            # cv2.rectangle(stamina_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            print("x,y,w,h:", x, y, w, h)
            ROI = stamina_image[y:y+h, x:x+w]

            digitFeatures(ROI)

    # plt.imshow(mask)
    # plt.show()
    # plt.imshow(result)
    # plt.show()


def digitFeatures(digit: np.array):
    baseDir = "./numbers"
    os.listdir(baseDir)
    plt.figure()
    plt.imshow(digit)
    roi = cv2.resize(digit, (57, 88))
    print("Please enter the number shown in the image after you close it: ")
    plt.show()
    number = input()
    numberDir = os.path.join(baseDir, number)
    numberLen = str(len(os.listdir(numberDir)))
    numberName = os.path.join(numberDir, numberLen)

    cv2.imwrite("{}.png".format(numberName), roi)


if __name__ == "__main__":

    # Load in base truth/reference images
    files = load.findFiles("../heroes/*jpg")
    baseImages = []
    for i in files:
        hero = cv2.imread(i)
        name = os.path.basename(i)
        baseImages.append((name, hero))

    # load in screenshot of heroes
    stamina_image = cv2.imread("./stamina-2.jpg")
    # plt.imshow(stamina_image)
    # plt.show()
    heroesDict = processing.getHeroes(stamina_image)
    print(len(heroesDict))

    cropHeroes = load.crop_heroes(heroesDict)
    print(len(cropHeroes))
    imageDB = load.build_flann(baseImages)

    for k, v in cropHeroes.items():

        name, baseHeroImage = imageDB.search(v, display=True)
        heroesDict[k]["label"] = name

    rows = generate_rows(heroesDict)
    staminaAreas = get_stamina_area(rows, heroesDict, stamina_image)
    get_text(staminaAreas)
