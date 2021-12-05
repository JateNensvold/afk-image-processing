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


def cachedproperty(func):
    """
        Used on methods to convert them to methods that replace themselves
        with their return value once they are called.
    """
    if func.__name__ in GV.CACHED:
        return GV.CACHED[func.__name__]

    def cache(*args):
        result = func(*args)
        GV.CACHED[func.__name__] = result
        return result
    return cache


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
    numbersFolder = GV.DATABASE_STAMINA_TEMPLATES_DIR

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
        result[mask == 0] = (255, 255, 255)

        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.threshold(
            result, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

        digit_contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_contours = imutils.grab_contours(digit_contours)
        digit_contours = imutils.contours.sort_contours(
            digit_contours,
            method="left-to-right")[0]
        digitText = []
        for digit in digit_contours:
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


@ cachedproperty
def signature_template_mask(templates: dict):
    siFolders = os.listdir(GV.SI_TEMPLATE_DIR)
    si_dict = {}

    for folder in siFolders:
        SIDir = os.path.join(GV.SI_TEMPLATE_DIR, folder)
        SIPhotos = os.listdir(SIDir)
        if folder == "40":
            continue
        for imageName in SIPhotos:

            siImage = templates[folder]["image"]

            templateImage = templates[folder].get(
                "crop", templates[folder].get("image"))
            mask = np.zeros_like(templateImage)

            if "morph" in templates[folder] and templates[folder]["morph"]:

                hsv_range = [np.array([0, 0, 206]), np.array([159, 29, 255])]

                thresh = processing.blur_image(
                    templateImage, hsv_range=hsv_range, reverse=True)

            else:
                templateGray = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)

                thresh = cv2.threshold(
                    templateGray, 0, 255,
                    cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

            inverted = cv2.bitwise_not(thresh)
            x, y = inverted.shape[:2]
            cv2.rectangle(inverted, (0, 0), (y, x), (255, 0, 0), 1)

            if folder == "0" or folder == "10":
                pass
            else:
                inverted = cv2.bitwise_not(inverted)

            siCont = cv2.findContours(
                inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            siCont = imutils.grab_contours(siCont)
            if folder == "0":

                siCont = sorted(siCont, key=cv2.contourArea, reverse=True)[
                    1:templates[folder]["contourNum"]+1]
            else:
                siCont = sorted(siCont, key=cv2.contourArea, reverse=True)[
                    :templates[folder]["contourNum"]]
            cv2.fillPoly(mask, siCont, [255, 255, 255])

            if folder not in si_dict:
                si_dict[folder] = {}
            # si_dict[folder]["image"] = SIGray
            si_dict[folder]["template"] = templateImage
            si_dict[folder]["source"] = siImage
            si_dict[folder]["mask"] = mask

    return si_dict


def signatureItemFeatures(hero: np.array,
                          si_dict: dict):
    """
    Runs template matching SI identification against the 'hero' passed in.
        When lvlRatioDict is passed in the templates will be rescaled to
        attempt and find the best template size for detecting SI objects

    Args:
        hero: np.array(x,y.3) representing an rgb image
        templates: dictionary of information about each SI template to get ran
            against the image
    Returns:
        dictionary with best "score" that each template achieved on the 'hero'
            image
    """
    hero_width, hero_height, _ = hero.shape

    x_div = 2.4
    y_div = 2.0
    hero_copy = hero.copy()

    crop_hero = hero[0: int(hero_height/y_div), 0: int(hero_width/x_div)]
    numberScore = {}

    # si_name_size = {"30": {"lower": 0.35, "upper": 0.55},
    #                 "20": {"lower": 0.35, "upper": 0.55},
    #                 "10": {"lower": 0.35, "upper": 0.55},
    #                 "0": {"lower": 0.35, "upper": 0.55}}

    # base_height

    # 30%
    start_percent = 20
    # 50%
    end_percent = 50

    for folder_name, imageDict in si_dict.items():
        si_image = imageDict["template"]
        # si_height, si_width = si_image.shape[:2]
        # sourceSIImage = imageDict["source"]

        for height_percent in range(start_percent, end_percent, 2):
            # si_image = imageDict["template"]
            height_percent = height_percent/100
            # si_height, original_width = si_image.shape[:2]
            new_si_height = round(height_percent * hero_height)
            new_si_width = round(height_percent * hero_width)
            # resize_height`
            # base_new_height = round(
            #     si_name_size[folder_name]["height"]) + pixel_offset
            # new_height = round(base_new_height * base_height_ratio)
            # scale_ratio = new_height/si_height
            # new_width = round(original_width * scale_ratio)

            si_image = cv2.resize(
                si_image, (new_si_width, new_si_height))
            si_image_gray = cv2.cvtColor(si_image, cv2.COLOR_BGR2GRAY)
            hero_gray = cv2.cvtColor(crop_hero, cv2.COLOR_BGR2GRAY)

            mask = cv2.resize(
                imageDict["mask"], (new_si_width, new_si_height))
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask_gray = mask

            # height, width = si_image.shape[:2]

            # sizedROI = cv2.resize(
            #     hero, (int(x * image_ratio), int(y * image_ratio)))
            if folder_name != "0":
                mask_gray = cv2.bitwise_not(mask_gray)
            # if folder_name == "10":

            try:
                try:
                    templateMatch = cv2.matchTemplate(
                        hero_gray, si_image_gray, cv2.TM_CCOEFF_NORMED,
                        mask=mask_gray)
                except Exception:

                    if crop_hero.shape[0] < si_image_gray.shape[0] or \
                            crop_hero.shape[1] < si_image_gray.shape[1]:
                        _height, _width = si_image_gray.shape[:2]
                        crop_hero = hero[0: max(int(hero_height/y_div),
                                                int(_height*1.2)),
                                         0: max(int(hero_width/x_div),
                                                int(_width*1.2))]
                        hero_gray = cv2.cvtColor(crop_hero, cv2.COLOR_BGR2GRAY)
                    templateMatch = cv2.matchTemplate(
                        hero_gray, si_image_gray, cv2.TM_CCOEFF_NORMED,
                        mask=mask_gray)
                (_, score, _, scoreLoc) = cv2.minMaxLoc(templateMatch)
                height, width = si_image.shape[:2]
                coords = (scoreLoc[0] + width, scoreLoc[1] + height)
            except Exception as e:
                raise IndexError(
                    "Si Size({}) is larger than source image({}),"
                    " check if source image is valid".format(
                        si_image.shape[:2], hero.shape[:2])) from e

            if folder_name not in numberScore:
                numberScore[folder_name] = []
            numberScore[folder_name].append(
                (score, height_percent, (scoreLoc, coords)))
    best_score = {}
    for _folder, _si_scores in numberScore.items():
        numberScore[_folder] = sorted(_si_scores, key=lambda x: x[0])
        _best_match = numberScore[_folder][-1]
        _score_loc = _best_match[2][0]
        _coords = _best_match[2][1]
        if GV.VERBOSE_LEVEL >= 1:
            cv2.rectangle(hero_copy, _score_loc, _coords, (255, 0, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(
                hero_copy, _folder, _coords, font, fontScale, color, thickness,
                cv2.LINE_AA)
        best_score[_folder] = round(_best_match[0], 3)
    # print(best_score)
    # load.display_image(hero_copy, display=True)
    return best_score


@ cachedproperty
def _furniture_template_mask(templates: dict):

    fi_dict = {}
    fi_folders = os.listdir(GV.FI_TEMPLATE_DIR)

    for folder in fi_folders:
        fi_dir = os.path.join(GV.FI_TEMPLATE_DIR, folder)
        fi_photos = os.listdir(fi_dir)
        for image_name in fi_photos:
            fi_image = templates[folder]["image"]
            template_image = templates[folder].get(
                "crop", templates[folder]["image"])
            mask = np.zeros_like(template_image)

            if "morph" in templates[folder] and templates[folder]["morph"]:
                se = np.ones((2, 2), dtype='uint8')
                # inverted = cv2.bitwise_not(inverted)

                lower = np.array([0, 8, 0])
                upper = np.array([179, 255, 255])
                hsv = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)
                thresh = cv2.inRange(hsv, lower, upper)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)
                thresh = cv2.bitwise_not(thresh)

            else:
                template_gray = cv2.cvtColor(
                    template_image, cv2.COLOR_BGR2GRAY)

                thresh = cv2.threshold(
                    template_gray, 147, 255,
                    cv2.THRESH_BINARY)[1]
                inverted = thresh
            x, y = inverted.shape[:2]
            cv2.rectangle(inverted, (0, 0), (y, x), (255, 0, 0), 1)
            inverted = cv2.bitwise_not(inverted)

            fi_contours = cv2.findContours(
                inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            fi_contours = imutils.grab_contours(fi_contours)
            fi_contours = sorted(fi_contours, key=cv2.contourArea,
                                 reverse=True)[:templates[
                                     folder]["contourNum"]]
            cv2.drawContours(mask, fi_contours, -1,
                             (255, 255, 255), thickness=cv2.FILLED)

            if folder not in fi_dict:
                fi_dict[folder] = {}
            # si_dict[folder][imageName] = siImage
            fi_dict[folder]["template"] = template_image
            fi_dict[folder]["source"] = fi_image

            fi_dict[folder]["mask"] = mask

    return fi_dict


# Deprecated function, functionality replaced with Yolov5 model at
#   image_processing/afk/fi/fi_detection/data/models
def _furnitureItemFeatures(hero: np.array, fi_dict: dict,
                           lvlRatioDict: dict = None):
    """
    Runs template matching FI identification against the 'hero' passed in.
        When lvlRatioDict is passed in the templates will be rescaled to
        attempt and find the best template size for detecting FI objects

    Args:
        hero: np.array(x,y.3) representing an rgb image
        fi_dict: dictionary of information about each FI template to get ran
            against the image
        lvlRatioDict: dictionary that contains the predicted height of each
            signature item based on precomputed text to si scaling calculations
    Returns:
        dictionary with best "score" that each template achieved on the 'hero'
            image
    """
    # variable_multiplier =
    x, y, _ = hero.shape
    x_div = 2.4
    y_div = 2.0
    x_offset = int(x*0.1)
    y_offset = int(y*0.30)
    # hero_copy = hero.copy()

    crop_hero = hero[y_offset: int(y*0.6), x_offset: int(x*0.4)]

    numberScore = {}
    neighborhood_size = 7
    sigmaColor = sigmaSpace = 75.

    # size_multiplier = 4

    old_crop_hero = crop_hero
    crop_hero = cv2.bilateralFilter(
        crop_hero, neighborhood_size, sigmaColor, sigmaSpace)

    # rgb_range = [
    #     np.array([190, 34, 0]), np.array([255, 184, 157])]
    # rgb_range = [
    #     np.array([180, 0, 0]), np.array([255, 246, 255])]

    # default_blur = processing.blur_image(
    #     crop_hero, rgb_range=rgb_range)
    # # (RMin = 180 , G
    # hero_crop_contours = cv2.findContours(
    #     default_blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # hero_crop_contours = imutils.grab_contours(hero_crop_contours)

    # # TODO add contour detection instead of template matching for low
    #   resolution images
    # hsv_range = [
    #     np.array([0, 50, 124]), np.array([49, 255, 255])]
# (RMin = 133 , GMin = 61, BMin = 35), (RMax = 255 , GMax = 151, BMax = 120)
    rgb_range = [np.array([133, 61, 35]), np.array([255, 151, 120])]

    blur_hero = processing.blur_image(crop_hero, rgb_range=rgb_range)

    fi_color_contours = cv2.findContours(
        blur_hero, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    fi_color_contours = imutils.grab_contours(fi_color_contours)
    fi_color_contours = sorted(fi_color_contours, key=cv2.contourArea,
                               reverse=True)[:2]
    # # # for fi_color_contour in fi_color_contour:
    # blur_hero_mask = np.zeros_like(blur_hero)
    crop_hero_mask = np.zeros_like(crop_hero)
    # hsv_rgb_mask = np.zeros_like(blur_hero)

    # master_contour = [
    #     _cont for _cont_list in fi_color_contours for _cont in _cont_list]
    # hull = cv2.convexHull(np.array(master_contour))

    # cv2.drawContours(crop_hero_mask, [
    #     hull], -1, (255, 255, 255), thickness=cv2.FILLED)

    for _cont in fi_color_contours:
        hull = cv2.convexHull(np.array(_cont))
        cv2.drawContours(crop_hero_mask, [
            hull], -1, (255, 255, 255), thickness=cv2.FILLED)
        # cv2.drawContours(crop_hero_mask, [_cont], -1, (255, 0, 0))

    # cv2.drawContours(blur_hero_mask, fi_color_contours, -1,
    #                  (255, 255, 255))
    # #  , thickness=cv2.FILLED)
    # # cv2.drawContours(crop_hero_mask, hero_crop_contours, -1,
    # #                  (255, 255, 255))
    # #  , thickness=cv2.FILLED)

    # cv2.drawContours(hsv_rgb_mask, fi_color_contours, -1,
    #                  (255, 255, 255), thickness=cv2.FILLED)
    # cv2.drawContours(hsv_rgb_mask, hero_crop_contours, -1,
    #                  (255, 255, 255), thickness=cv2.FILLED)

    # # color_mask = cv2.merge(
    # #     [blur_hero, blur_hero, blur_hero])
    blur_hero = cv2.bitwise_and(crop_hero_mask, crop_hero)
    blur_hero[np.where((blur_hero == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    old_crop_hero = cv2.bitwise_and(crop_hero_mask, old_crop_hero)

    # blur_hero = crop_hero
    for pixel_offset in range(-5, 15, 1):

        for folder_name, imageDict in fi_dict.items():
            fi_image = imageDict["template"]

            sourceSIImage = imageDict["source"]
            hero_h, hero_w = sourceSIImage.shape[:2]

            original_height, original_width = fi_image.shape[:2]

            base_height_ratio = original_height/hero_h
            # resize_height
            base_new_height = max(round(
                lvlRatioDict[folder_name]["height"]), 12)+pixel_offset
            # base_new_height = round(lvlRatioDict[folder_name]["height"])

            new_height = round(base_new_height * base_height_ratio)
            scale_ratio = new_height/original_height
            new_width = round(original_width * scale_ratio)
            fi_image = cv2.resize(
                fi_image, (new_width, new_height))
            # fi_gray = cv2.cvtColor(fi_image, cv2.COLOR_BGR2GRAY)

            #   Min = 0, BMin = 0), (RMax = 255 , GMax = 246, BMax = 255)

            mask = cv2.resize(
                imageDict["mask"], (new_width, new_height))
            # mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # hero_gray = cv2.cvtColor(crop_hero, cv2.COLOR_BGR2GRAY)

            height, width = fi_image.shape[:2]

            # sizedROI = cv2.resize(
            #     hero, (int(x * image_ratio), int(y * image_ratio)))
            # if folder_name != "0":
            #     mask_gray = cv2.bitwise_not(mask_gray)

            try:
                templateMatch = cv2.matchTemplate(
                    blur_hero, fi_image, cv2.TM_CCOEFF_NORMED,
                    mask=mask)
            except Exception:
                if blur_hero.shape[0] < fi_image.shape[0] or \
                        blur_hero.shape[1] < fi_image.shape[1]:
                    _height, _width = fi_image.shape[:2]
                    blur_hero = hero[
                        y_offset: max(int(y/y_div),
                                      int(_height * 1.2)+y_offset),
                        x_offset: max(int(x/x_div),
                                      int(_width * 1.2)+x_offset), ]
                    # blur_hero = crop_hero
                    # hero_gray = cv2.cvtColor(crop_hero, cv2.COLOR_BGR2GRAY)
                templateMatch = cv2.matchTemplate(
                    blur_hero, fi_image, cv2.TM_CCOEFF_NORMED,
                    mask=mask)

            (_, score, _, scoreLoc) = cv2.minMaxLoc(templateMatch)
            scoreLoc = (scoreLoc[0], scoreLoc[1])
            coords = (scoreLoc[0] + width, scoreLoc[1] + height)

            if folder_name not in numberScore:
                numberScore[folder_name] = []
            numberScore[folder_name].append(
                (score, pixel_offset, (scoreLoc, coords)))
    best_score = {}
    import math
    for _folder, _fi_scores in numberScore.items():
        numberScore[_folder] = sorted(_fi_scores, key=lambda x: x[0])
        _t = [_num for _num in numberScore[_folder]
              if not math.isinf(_num[0])]
        if len(_t) == 0:
            if GV.VERBOSE_LEVEL >= 1:
                print("Failed to find FI", _folder,  numberScore[_folder])
            _best_match = (0, 0, ((0, 0), (0, 0)))
        else:
            _best_match = _t[-1]

        _score_loc = _best_match[2][0]
        _coords = _best_match[2][1]
        cv2.rectangle(blur_hero, _score_loc, _coords, (255, 0, 0), 1)
        best_score[_folder] = _best_match[0]
        print(best_score[_folder])

    return best_score


def digitFeatures(digit: np.array, save_dir=None):
    """
    Save a presized digit to whatever number is entered
    Args:
        digit: presized image of a digit, that will be saved as a training
            template under whatever digitName/label is entered when prompted
            by terminal
    Return:
        None
    """

    base_dir = GV.DATABASE_STAMINA_TEMPLATES_DIR
    if save_dir:
        base_dir = save_dir
    digitFolders = os.listdir(base_dir)
    plt.figure()
    plt.imshow(digit)
    plt.ion()

    plt.show()

    number = input("Please enter the number shown in the image: ")

    plt.close()

    if number not in digitFolders:
        print("No such folder {}".format(number))
        number = "none"

    numberDir = os.path.join(base_dir, number)
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
    heroesDict, rows = processing.getHeroes(stamina_image)

    cropHeroes = load.crop_heroes(heroesDict)
    imageDB = load.build_flann(baseImages)

    for k, v in cropHeroes.items():

        hero_info, baseHeroImage = imageDB.search(v, display=False)
        heroesDict[k]["label"] = hero_info.name

    staminaAreas = get_stamina_area(rows, heroesDict, stamina_image)
    staminaOutput = get_text(staminaAreas)
    output = {}

    for name, text in staminaOutput.items():
        label = heroesDict[name]["label"]
        if label not in output:
            output[label] = {}
        output[label]["stamina"] = text

    outputJson = json.dumps(output)
