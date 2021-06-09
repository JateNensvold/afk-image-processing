import cv2
import os
import image_processing.build_db as BD
import image_processing.globals as GV
import image_processing.load_images as load
import image_processing.processing as processing
import image_processing.stamina as stamina
import collections


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

    baseImages["0"]["image"] = image_0
    baseImages["0"]["contourNum"] = 2

    x, y, _ = image_10.shape
    newx = int(x*0.6)
    newy = int(y*0.6)
    baseImages["10"]["image"] = image_10
    baseImages["10"]["crop"] = image_10[0:newy, 0:newx]
    baseImages["10"]["contourNum"] = 2
    baseImages["10"]["morph"] = True

    x, y, _ = image_20.shape
    newx = int(x*0.3)
    newy = int(y*0.7)
    starty = int(y*0.6)
    baseImages["20"]["image"] = image_20
    baseImages["20"]["crop"] = image_20[starty:, 0:x]
    baseImages["20"]["contourNum"] = 2

    x, y, _ = image_30.shape
    newx = int(x*0.3)
    newy = int(y*0.7)
    baseImages["30"]["image"] = image_30
    baseImages["30"]["crop"] = image_30[0:newy, 0:newx]
    baseImages["30"]["contourNum"] = 1

    for name, imageDict in baseImages.items():
        image = imageDict["image"]
        crop = False
        if "crop" in imageDict:
            crop = True
            image = [imageDict["image"], imageDict["crop"]]
        load.display_image(image, multiple=crop)

    siPath = GV.siPath

    hero_ss = cv2.imread("../image0.png")
    heroesDict = processing.getHeroes(hero_ss)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    circle_fail = 0

    for k, v in heroesDict.items():
        name, baseHeroImage = imageDB.search(v["image"], display=False)
        heroesDict[k]["label"] = name
        si_scores = stamina.signatureItemFeatures(v["image"], baseImages)
        x = heroesDict[k]["dimensions"]["x"]
        y = heroesDict[k]["dimensions"]["y"]
        if si_scores == -1:
            circle_fail += 1
            best_si = "none"
        else:

            best_si = max(si_scores, key=si_scores.get)
        coords = (x[0], y[0])
        name = "{},{}".format(name, best_si)
        print(best_si)
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

    # si_30 = load.findFiles(si_x_folder)
    # si_30_images = []
    # for i in si_30:
    #     hero = cv2.imread(i)
    #     baseName = os.path.basename(i)
    #     name, ext = os.path.splitext(baseName)

    # si_30_images.append((name, hero))
    avg_0 = 0
    avg_10 = 0
    avg_20 = 0
    avg_30 = 0

    # numImages = len(si_30_images)
    results = {"0": 0, "10": 0, "20": 0, "30": 0}

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
    # print("0", results["0"]/len(si_30_images))
    # print("10", results["10"]/len(si_30_images))
    # print("20", results["20"]/len(si_30_images))
    # print("30", results["30"]/len(si_30_images))
    # print("circle fails", circle_fail)
