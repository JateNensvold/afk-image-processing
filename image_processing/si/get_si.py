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
import multiprocessing
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# from detectron2.checkpoint import DetectionCheckpointer

# import PIL
import torch
# import psutil
import warnings
warnings.filterwarnings("ignore")
MODEL = None
BORDER_MODEL = None


def get_si(roster_image, image_name, debug_raw=False, imageDB=None,
           hero_dict=None):
    if imageDB is None:
        imageDB = BD.get_db(enrichedDB=True)
    # hero_ss = image

    baseImages = collections.defaultdict(dict)

    image_0 = cv2.imread(os.path.join(GV.siBasePath, "0", "0.png"))
    image_10 = cv2.imread(os.path.join(GV.siBasePath, "10", "10.png"))
    image_20 = cv2.imread(os.path.join(GV.siBasePath, "20", "20.png"))
    image_30 = cv2.imread(os.path.join(GV.siBasePath, "30", "30.png"))

    fi_3_image = cv2.imread(os.path.join(GV.fi_base_path, "3", "3fi.png"))
    fi_9_image = cv2.imread(os.path.join(GV.fi_base_path, "9", "9fi.png"))

    baseImages["3"]["image"] = fi_3_image
    baseImages["3"]["crop"] = fi_3_image[0:, 0:]
    baseImages["3"]["contourNum"] = 1

    baseImages["9"]["image"] = fi_9_image
    baseImages["9"]["crop"] = fi_9_image[0:, 0:]
    baseImages["9"]["contourNum"] = 1

    baseImages["0"]["image"] = image_0
    baseImages["0"]["contourNum"] = 2
    # baseImages["0"]["height"] = 52.6
    # baseImages["0"]["width"] = 52.06666666666667

    # x, y, _ = image_10.shape
    # new_x = int(x*0.7)
    # new_y = int(y*0.6)
    baseImages["10"]["image"] = image_10
    baseImages["10"]["crop"] = image_10[0:, 0:]
    baseImages["10"]["contourNum"] = 3
    baseImages["10"]["morph"] = True
    # baseImages["10"]["height"] = 63
    # baseImages["10"]["width"] = 63.52173913043478

    x, y, _ = image_20.shape

    # start_y = int(y*0.45)
    end_x = int(x*0.5)
    baseImages["20"]["image"] = image_20
    baseImages["20"]["crop"] = image_20[0:, 0:end_x]
    baseImages["20"]["contourNum"] = 2
    # baseImages["20"]["height"] = 75.1891891891892
    # baseImages["20"]["width"] = 64.27027027027027

    # new_x = int(x*0.5)
    # new_y = int(y*0.2)
    # baseImages["30"]["image"] = image_30
    # baseImages["30"]["crop"] = image_30[0:new_y, 0:]
    # baseImages["30"]["contourNum"] = 1
    x, y, _ = image_30.shape
    new_x = int(x*0.3)
    new_y = int(y*0.7)
    baseImages["30"]["image"] = image_30
    baseImages["30"]["crop"] = image_30[0:new_y, 0:new_x]
    baseImages["30"]["contourNum"] = 1
    # baseImages["30"]["height"] = 78.23076923076923
    # baseImages["30"]["width"] = 79.94871794871794

    csvfile = open(
        "/home/nate/projects/afk-image-processing/image_processing/scripts/"
        "lvl_txt_si_scale.txt",
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
        roster_image, blur_args=blur_args)

    if hero_dict is not None:
        hero_dict["hero_dict"] = heroesDict
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
        graded_avg_bin[si_name]["height"] = frequency_height_adjust

    si_dict = stamina.signature_template_mask(baseImages)
    # fi_dict = stamina.furniture_template_mask(baseImages)

    # opencv_model = cv2.dnn.readNetFromONNX(
    #     "/home/nate/projects/afk-image-processing/image_processing/fi/"
    #     "fi_detection/yolov5/yolov5s.onnx")

    # fi_model = rt.InferenceSession(
    #     "/home/nate/projects/afk-image-processing/image_processing/fi/"
    #     "fi_detection/yolov5/yolov5s.onnx", None)


# names: ['1 star', '2 star', '3 fi', '3 star', '4 star', '5 star', '9 fi']
    model_labels = ["A1", "A2", "3", "A3", "A4", "A5", "9"]
    # ascended', 'elite', 'elite+', 'legendary', 'legendary+', 'mythic',
    # 'mythic+
    border_labels = ["B", "E", "E+", "L", "L+", "M", "M+", "A"]
    model_dict = {"labels": model_labels, "border_labels": border_labels}

    if not GV.DEBUG and GV.PARALLEL:
        pool = multiprocessing.Pool()

        all_args = [({"name": _hero_name,
                      "info": _hero_info_dict,
                      "si_dict": si_dict,
                      "graded_avg_bin": graded_avg_bin,
                      "model": model_dict}
                     )for _hero_name, _hero_info_dict in heroesDict.items()]

        reduced_values = pool.map(parallel_detect, all_args)
    else:
        reduced_values = []
        for _hero_name, _hero_info_dict in heroesDict.items():

            results = parallel_detect(
                {"name": _hero_name,
                 "info": _hero_info_dict,
                 "si_dict": si_dict,
                 "graded_avg_bin": graded_avg_bin,
                 "model": model_dict})

            reduced_values.append(results)

    return_dict = {}

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 3
    color = (255, 255, 0)
    # color = (0,0,0)
    thickness = 2
    fontScale = 0.5 * (rows.get_avg_width()/100)

    hero_count = 0
    for _hero_data in reduced_values:
        # result = _hero_data["result"]
        name = _hero_data["pseudo_name"]
        hero_name, _ = imageDB.search(heroesDict[name]["image"])
        _hero_data["result"].insert(0, hero_name)
        # result = "{} {}".format(hero_name, result)
        # _hero_data["result"] = result

        # if hero_name != "food":
        return_dict[name] = _hero_data
        if GV.VERBOSE_LEVEL >= 1:
            result = str(_hero_data["result"])
            coords = _hero_data["coords"]
            text_size = cv2.getTextSize(result, font, fontScale, thickness)
            height = text_size[0][1]
            coords = (coords[0], coords[1] + round(5 * height))

            cv2.putText(
                roster_image, result, coords, font, abs(
                    fontScale), color, thickness,
                cv2.LINE_AA)
        if hero_dict is not None:
            heroesDict["{}_{}".format(
                "".join(_hero_data["result"]), hero_count)] = heroesDict[name]
        hero_count += 1
    json_dict = {}
    json_dict[image_name] = {}
    json_dict[image_name]["rows"] = len(rows)
    json_dict[image_name]["columns"] = max([len(_row) for _row in rows])

    json_dict[image_name]["heroes"] = []

    for _row in rows:
        temp_list = []
        for _row_item in _row:
            hero_data = return_dict[_row_item.name]["result"]
            if hero_data[0] != "food":
                if debug_raw:
                    _raw_score = return_dict[_row_item.name]["score"]
                    hero_data.append(_raw_score)
                temp_list.append(hero_data)
        json_dict[image_name]["heroes"].append(temp_list)
    # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    return json_dict


def parallel_detect(info_dict):
    k = info_dict["name"]
    v = info_dict["info"]
    si_dict = info_dict["si_dict"]
    # fi_dict = info_dict["fi_dict"]

    global MODEL
    if not MODEL:
        MODEL = torch.hub.load(
            '/ home/nate/projects/afk-image-processing/image_processing/fi /" \
                               "fi_detection/yolov5',
            "custom",
            "/home/nate/projects/afk-image-processing/image_processing/fi/"
            "fi_detection/yolov5/runs/train/yolov5s_results_v22/"
            "weights/best.pt",
            source="local",
            force_reload=True,
            verbose=False)

    graded_avg_bin = info_dict["graded_avg_bin"]
    # for k, v in heroes_dict.items():
    # heroes_dict[k]["label"] = name
    return_dict = {}
    si_scores = stamina.signatureItemFeatures(
        v["image"], si_dict, graded_avg_bin)
    # fi_scores = stamina.furnitureItemFeatures(
    #     v["image"], fi_dict, graded_avg_bin)
    x = v["object"].dimensions.x
    y = v["object"].dimensions.y
    # if fi_scores["9"] == 0 or fi_scores["3"] == 0:
    #     return_dict["display"] = True
    # if fi_scores["9"] > 0.7:
    #     best_fi = "9"
    # elif fi_scores["3"] > 0.5:
    #     best_fi = "3"
    # else:
    #     best_fi = "0"
    test_img = v["image"]
    model_image_size = (416, 416)
    test_img = cv2.resize(
        test_img,
        model_image_size,
        interpolation=cv2.INTER_CUBIC)
    test_img = test_img[..., ::-1]
    results = MODEL([test_img], size=416)

    model_labels = info_dict["model"]["labels"]
    border_labels = info_dict["model"]["border_labels"]

    results_array = results.pandas().xyxy[0]
    RA = results_array
    # fi_filtered_results = results_array.query('class == 2 or class == 6')
    fi_filtered_results = RA.loc[(RA['class'] == 2) |
                                 (RA['class'] == 6)]

    star_filtered_results = RA.loc[(RA['class'] != 2) &
                                   (RA['class'] != 6)]
    # star_filtered_results = results_array.query('class != 2 and class != 6')

    if len(fi_filtered_results) > 0:
        fi_final_results = fi_filtered_results.sort_values(
            "confidence").iloc[0]
        best_fi = model_labels[fi_final_results["class"]]

        fi_scores = {best_fi:
                     fi_final_results["confidence"]}
        if fi_final_results["confidence"] < 0.8:
            best_fi = "0"
            fi_scores = {best_fi: 1.0}
    else:
        best_fi = "0"
        fi_scores = {best_fi: 1.0}
    star = False
    if len(star_filtered_results) > 0:
        final_star_results = star_filtered_results.sort_values(
            "confidence", ascending=False).iloc[0]
        best_ascension = model_labels[final_star_results["class"]]

        ascension_scores = {best_ascension:
                            final_star_results["confidence"]}
        if final_star_results["confidence"] > 0.8:
            star = True
    if not star:
        global BORDER_MODEL
        if not BORDER_MODEL:
            cfg = get_cfg()

            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.DATASETS.TRAIN = ("border_dataset_train",)
            cfg.DATASETS.TEST = ("border_dataset_val",)

            cfg.DATALOADER.NUM_WORKERS = 0
            # Let training initialize from model zoo
            # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            #     "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.WEIGHTS = (
                "detectron2://COCO-Detection/"
                "faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl")

            cfg.SOLVER.IMS_PER_BATCH = 2
            cfg.SOLVER.BASE_LR = 0.00025

            # adjust up if val mAP is still rising, adjust down if overfit
            cfg.SOLVER.MAX_ITER = 3000

            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8

            # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
            cfg.MODEL.WEIGHTS = ("/home/nate/projects/afk-image-processing/"
                                 "image_processing/fi/fi_detection/"
                                 "hero_border_model_output/model_final.pth")
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(border_labels)
            BORDER_MODEL = DefaultPredictor(cfg)

        raw_border_results = BORDER_MODEL(test_img)
        border_results = raw_border_results["instances"]
        # print(border_results)
        classes = border_results.pred_classes.cpu().tolist()
        scores = border_results.scores.cpu().tolist()
        best_class = list(zip([border_labels[class_num]
                          for class_num in classes], scores))[0]
        # # pred_class_names = list(map(lambda x: class_names[x], classes))
        # # print(pred_class_names)
        # scores = border_results["scores"]
        # pred_dict = zip(classes, scores)
        # print(pred_dict)
        # border_results_array = border_results.pandas().xyxy[0]
        # if len(border_results_array) > 0:
        #     final_border_results = border_results_array.sort_values(
        #         "confidence", ascending=False).iloc[0]
        #     best_ascension = border_labels[final_border_results["class"]]
        #     ascension_scores = {best_ascension:
        #                         final_border_results["confidence"]}

        #     if final_border_results["confidence"] < 0.8:
        #         best_ascension = " "
        #         ascension_scores = {best_ascension: 1.0}
        # else:
        best_ascension = best_class[0]
        ascension_scores = {best_ascension: best_class[1]}

    if si_scores == -1:
        # circle_fail += 1
        best_si = "none"
    else:
        if si_scores["30"] > 0.6:
            best_si = "30"
        elif si_scores["20"] > 0.7:
            best_si = "20"
        elif si_scores["10"] > 0.55:
            best_si = "10"
        elif si_scores["10"] < 0.55 and si_scores["0"] > 0.75:
            best_si = "00"
        elif si_scores["10"] > 0.50:
            best_si = "10"
        else:
            # si_label_list = ["0", "10"]
            # # key=lambda x: heroes[x[0][1]]["dimensions"]["y"][0]
            # best_si = max(si_label_list, key=lambda x: si_scores[x])
            # best_si_score = si_scores[best_si]
            # if best_si_score < 0.4:
            #     # best_si = "n/a"
            best_si = "00"

    coords = (x, y)
    # name = "{},{}".format(name, best_si)
    # name = "{} s:{:.3}".format(best_fi, fi_score)
    name = [best_si, best_fi, best_ascension]
    # "{}{}".format(best_si, best_fi)
    # print("pid: {} name:{} stat:{}".format(
    #     multiprocessing.current_process().pid, k, name))
    return_dict["si"] = best_si
    # return_dict["fi"] = best_fi
    return_dict["score"] = {}
    return_dict["score"]["si"] = si_scores
    return_dict["score"]["fi"] = fi_scores
    return_dict["score"]["ascension"] = ascension_scores

    return_dict["result"] = name
    return_dict["pseudo_name"] = k
    return_dict["coords"] = coords
    # return_dict["name"] = hero
    # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    return return_dict


if __name__ == "__main__":
    # pool = multiprocessing.Pool()
    json_dict = get_si(GV.image_ss, GV.image_ss_name, debug_raw=True)
    if GV.VERBOSE_LEVEL >= 1:
        load.display_image(GV.image_ss, display=True)
    if GV.VERBOSE_LEVEL == 0:
        print("{{\"heroes\": {}}}".format(
            json_dict[GV.image_ss_name]["heroes"]))
    else:
        print(json_dict)
    # with open("temp.json", "w") as f:
    #     json.dump(json_dict, f)
