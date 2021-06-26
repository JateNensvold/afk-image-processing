import cv2
import os
import csv
import image_processing.build_db as BD
import image_processing.globals as GV
import image_processing.load_images as load
import image_processing.processing as processing
import image_processing.stamina as stamina
import collections
import numpy as np
import image_processing.scripts.getSISize as siScript


def rollingAverage(avg, newSample, size):
    avg -= avg / size
    avg += newSample/size
    return avg


if __name__ == "__main__":

    # imageDB = BD.buildDB(enrichedDB=True)
    imageDB = BD.buildDB(enrichedDB=True)

    baseImages = collections.defaultdict(dict)

    image_0 = cv2.imread(os.path.join(GV.siBasePath, "0", "0.png"))
    image_10 = cv2.imread(os.path.join(GV.siBasePath, "10", "10.png"))
    image_20 = cv2.imread(os.path.join(GV.siBasePath, "20", "20.png"))
    image_30 = cv2.imread(os.path.join(GV.siBasePath, "30", "30.png"))

    fi_base_images = collections.defaultdict(dict)

    fi_3_image = cv2.imread(os.path.join(GV.fi_base_path, "3", "3fi.png"))
    fi_9_image = cv2.imread(os.path.join(GV.fi_base_path, "9", "9fi.png"))

    baseImages["3"]["image"] = fi_3_image
    baseImages["3"]["crop"] = fi_3_image[0:, 0:]
    baseImages["3"]["contourNum"] = 2

    baseImages["9"]["image"] = fi_9_image
    baseImages["9"]["crop"] = fi_9_image[0:, 0:]
    baseImages["9"]["contourNum"] = 2

    baseImages["0"]["image"] = image_0
    baseImages["0"]["contourNum"] = 2
    # baseImages["0"]["height"] = 52.6
    # baseImages["0"]["width"] = 52.06666666666667

    x, y, _ = image_10.shape
    newx = int(x*0.7)
    newy = int(y*0.6)
    baseImages["10"]["image"] = image_10
    baseImages["10"]["crop"] = image_10[0:, 0:newx]
    baseImages["10"]["contourNum"] = 2
    baseImages["10"]["morph"] = True
    # baseImages["10"]["height"] = 63
    # baseImages["10"]["width"] = 63.52173913043478

    x, y, _ = image_20.shape

    starty = int(y*0.45)
    endx = int(x*0.5)
    baseImages["20"]["image"] = image_20
    baseImages["20"]["crop"] = image_20[0:, 0:endx]
    baseImages["20"]["contourNum"] = 2
    # baseImages["20"]["height"] = 75.1891891891892
    # baseImages["20"]["width"] = 64.27027027027027




    # newx = int(x*0.5)
    # newy = int(y*0.2)
    # baseImages["30"]["image"] = image_30
    # baseImages["30"]["crop"] = image_30[0:newy, 0:]
    # baseImages["30"]["contourNum"] = 1
    x, y, _ = image_30.shape
    newx = int(x*0.3)
    newy = int(y*0.7)
    baseImages["30"]["image"] = image_30
    baseImages["30"]["crop"] = image_30[0:newy, 0:newx]
    baseImages["30"]["contourNum"] = 1
    # baseImages["30"]["height"] = 78.23076923076923
    # baseImages["30"]["width"] = 79.94871794871794

    csvfile = open(
        "/home/nate/projects/afk-image-processing/image_processing/scripts/lvl_txt_si_scale.txt",
        "r")

    header = ["digitName", "si_name", "v_scale"]

    reader = csv.DictReader(csvfile, header)

    for row in reader:
        _digit_name = row["digitName"]
        _si_name = row["si_name"]
        _v_scale = float(row["v_scale"])
        if _digit_name not in baseImages[_si_name]:
            baseImages[_si_name][_digit_name] = {}
        baseImages[_si_name][_digit_name]["v_scale"] = _v_scale

    for name, imageDict in baseImages.items():
        image = imageDict["image"]
        crop = False
        if "crop" in imageDict:
            crop = True
            image = [imageDict["image"], imageDict["crop"]]
        # load.display_image(image, multiple=crop)

    siPath = GV.siPath

    # hero_ss = cv2.imread(
    #     "/home/nate/projects/afk-image-processing/test_2.jpg")
    hero_ss = GV.image_ss

    # (hMin = 0 , sMin = 68, vMin = 170), (hMax = 35 , sMax = 91, vMax = 255)
    # lower_hsv = np.array([0, 68, 170])
    # upper_hsv = np.array([35, 91, 255])

    # (hMin = 23 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)
    # lower_hsv = np.array([23, 0, 0])
    # upper_hsv = np.array([179, 255, 255])

    # (hMin = 12 , sMin = 75, vMin = 212), (hMax = 23 , sMax = 109, vMax = 253)
    # lower_hsv = np.array([12, 75, 212])
    # upper_hsv = np.array([23, 109, 253])

    # (hMin = 5 , sMin = 79, vMin = 211), (hMax = 21 , sMax = 106, vMax = 250)
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([179, 255, 192])

# (hMin = 0 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 192)

    hsv_range = [lower_hsv, upper_hsv]
    blur_args = {"hsv_range": hsv_range}
    heroesDict, rows = processing.getHeroes(
        hero_ss, si_adjustment=0, blur_args=blur_args)
    # import sys
    # sys.exit(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    circle_fail = 0

    digit_bins = {}
    for k, v in heroesDict.items():
        bins = siScript.getDigit(v["image"])
        v["digit_info"] = bins
        for digitName, tempDigitdict in bins.items():

            digitTuple = tempDigitdict["digit_info"]
            digitTop = digitTuple[0]
            digitBottom = digitTuple[1]
            digitHeight = digitBottom - digitTop
            if digitName not in digit_bins:
                digit_bins[digitName] = []
            digit_bins[digitName].append(digitHeight)
    avg_bin = {}
    total_digit_occurrences = 0
    for k, v in digit_bins.items():
        avg = np.mean(v)
        avg_bin[k] = {}
        avg_bin[k]["height"] = avg
        occurrence = len(v)
        avg_bin[k]["count"] = occurrence
        total_digit_occurrences += occurrence

        # print("{} {}".format(k, avg))
    graded_avg_bin = {}
    for si_name, image_dict in baseImages.items():
        if si_name not in graded_avg_bin:
            graded_avg_bin[si_name] = {}
        frequency_height_adjust = 0
        for digit_name, scale_dict in avg_bin.items():

            v_scale = baseImages[si_name][digit_name]["v_scale"]

            digit_count = scale_dict["count"]
            digit_height = scale_dict["height"]
            digit_freqency = digit_count / total_digit_occurrences

            frequency_height_adjust += (v_scale *
                                        digit_height) * digit_freqency
            # print(si_name, v_scale * digit_height)
            # graded_avg_bin[si_name]["height"] += graded_avg_bin[si_name][
            #     "height"] + frequency_height_adjust
        graded_avg_bin[si_name]["height"] = frequency_height_adjust

    # fi_graded_avg_bin = {}
    # for fi_name, fi_image_dict in fi_base_images.items():
    #     if si_name not in fi_graded_avg_bin:
    #         fi_graded_avg_bin[fi_name] = {}
    #     frequency_height_adjust = 0
    #     for digit_name, scale_dict in avg_bin.items():

    #         v_scale = fi_base_images[fi_name][digit_name]["v_scale"]

    #         digit_count = scale_dict["count"]
    #         digit_height = scale_dict["height"]
    #         digit_freqency = digit_count / total_digit_occurrences

    #         frequency_height_adjust += (v_scale *
    #                                     digit_height) * digit_freqency
    #         # print(si_name, v_scale * digit_height)
    #         # graded_avg_bin[si_name]["height"] += graded_avg_bin[si_name][
    #         #     "height"] + frequency_height_adjust
    #     fi_graded_avg_bin[si_name]["height"] = frequency_height_adjust

    # print(graded_avg_bin)
    # import sys
    # sys.exit()

    for k, v in heroesDict.items():
        name, baseHeroImage = imageDB.search(v["image"])
        heroesDict[k]["label"] = name
        # siScript.getLevelDigit(v["image"], ,train=False)

        si_scores = stamina.signatureItemFeatures(
            v["image"], baseImages, graded_avg_bin)
        fi_scores = stamina.furnitureItemFeatures(
            v["image"], baseImages, graded_avg_bin)
        x = heroesDict[k]["object"][0][0]
        y = heroesDict[k]["object"][0][1]
        if fi_scores["9"] > 0.6:
            best_fi = "9"
            fi_score = fi_scores["9"]
        elif fi_scores["3"] > 0.65:
            best_fi = "3"
            fi_score = fi_scores["3"]
        else:
            best_fi = "0"
            fi_score = fi_scores["3"]

        if si_scores == -1:
            circle_fail += 1
            best_si = "none"
        else:
            print(si_scores)
            if si_scores["30"] > 0.45:
                best_si = "30"
            elif si_scores["20"] > 0.6:
                best_si = "20"
            elif si_scores["10"] > 0.4:
                best_si = "10"
            else:
                si_label_list = ["0", "10"]
                # key=lambda x: heroes[x[0][1]]["dimensions"]["y"][0]
                best_si = max(si_label_list, key=lambda x: si_scores[x])
                best_si_score = si_scores[best_si]
                if best_si_score < 0.4:
                    # best_si = "n/a"
                    best_si = "00"

        coords = (x, y)
        # name = "{},{}".format(name, best_si)
        # name = "{} s:{:.3}".format(best_fi, fi_score)
        name = "{}{}".format(best_si, best_fi)

        print(best_si)
        # print(si_scores)
        cv2.putText(
            hero_ss, name, coords, font, fontScale, color, thickness,
            cv2.LINE_AA)
    load.display_image(hero_ss, display=True)

    # stamina_image = cv2.imread("./stamina.jpg")
    # heroesDict = processing.getHeroes(stamina_image)

    # cropHeroes = load.crop_heroes(heroesDict)
    # si_x_folder = os.path.join(siPath,  "train", "0", "*")
    # si_x_folder = os.path.join(siPath, "train", "10", "*")
    # si_x_folder = os.path.join(siPath,  "train", "20", "*")
    # si_x_folder = os.path.join(siPath,  "train", "30", "*")
    # print(si_x_folder)
    # si_30 = load.findFiles(si_x_folder)
    # print(si_30)

    # si_30_images = []
    # for i in si_30:
    #     hero = cv2.imread(i)
    #     baseName = os.path.basename(i)
    #     name, ext = os.path.splitext(baseName)

    #     si_30_images.append((name, hero))
    # avg_0 = 0
    # avg_10 = 0
    # avg_20 = 0
    # avg_30 = 0

    # # numImages = len(si_30_images)
    # results = {"0": [], "10": [], "20": [], "30": []}

    # for k, v in si_30_images:

    #     # name, baseHeroImage = imageDB.search(v, display=False)
    #     # if name not in results:
    #     #     results[name] = 0
    #     # results[name] = results[name] + 1
    #     si_scores = stamina.signatureItemFeatures(v, baseImages)
    #     if si_scores == -1:
    #         circle_fail += 1
    #         continue
    #     results["0"].append(si_scores["0"])
    #     results["10"].append(si_scores["10"])
    #     results["20"].append(si_scores["20"])
    #     results["30"].append(si_scores["30"])
    #     # avg_0 = rollingAverage(avg_0, si_scores["0"], numImages)
    #     # avg_10 = rollingAverage(avg_10, si_scores["10"], numImages)
    #     # avg_20 = rollingAverage(avg_20, si_scores["20"], numImages)
    #     # avg_30 = rollingAverage(avg_30, si_scores["30"], numImages)

    #     # heroesDict[k]["label"] = name
    # print(results)
    # print("0", sum(results["0"])/len(si_30_images),
    #       min(results["0"]), max(results["0"]))
    # print("10", sum(results["10"])/len(si_30_images),
    #       min(results["10"]), max(results["10"]))
    # print("20", sum(results["20"])/len(si_30_images),
    #       min(results["20"]), max(results["20"]))
    # print("30", sum(results["30"])/len(si_30_images),
    #       min(results["30"]), max(results["30"]))
    # print("circle fails", circle_fail)
