import time
import threading
import warnings
import cv2

import image_processing.build_db as BD
import image_processing.globals as GV
import image_processing.load_images as load
import image_processing.processing as processing
import image_processing.helpers.load_models as LM

import numpy as np

warnings.filterwarnings("ignore")


SI_LABELS = {
    0: "0",
    2: "10",
    4: "20",
    7: "30"
}

FI_LABELS = {
    5: "3",
    10: "9"
}

ASCENSION_STAR_LABELS = {
    1: "A1",
    3: "A2",
    6: "A3",
    8: "A4",
    9: "A5"
}


def get_si(roster_image, image_name, debug_raw=None, imageDB=None,
           hero_dict=None, faction=False):
    """
    Detect AFK Arena heroes from a roster screenshot and for each hero detect
        "FI", "SI", "Ascension", and "hero Name"
    Args:
        roster_image: image to run segmentation and detection on
        image_name: name of 'roster_image' all results in return dictionary
            are placed under this name in return dictionary
        debug_raw: flag to add raw values for SI, FI and Ascension detection
            to return dictionary
        imageDB: If this variable is None, imageDB is generated otherwise it
            is assumed imageDB is a fully initialized
            image_processing.database.imageDB.imageSearch object
        hero_dict: If this variable is not None, its assumed to be an empty
            dictionary that to return the hero segmentation dictionary
            detected from roster_image
        faction: flag to add faction output to hero feature list in the return
            dict
    """
    if debug_raw is None:
        if GV.VERBOSE_LEVEL >= 1:
            debug_raw = True
        else:
            debug_raw = False

    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([179, 255, 192])

    hsv_range = [lower_hsv, upper_hsv]
    blur_args = {"hsv_range": hsv_range}
    heroesDict, rows = processing.getHeroes(
        roster_image, blur_args=blur_args)

    if hero_dict is not None:
        hero_dict["hero_dict"] = heroesDict

    reduced_values = []
    for _hero_name, _hero_info_dict in heroesDict.items():

        results = detect_features(
            _hero_name, _hero_info_dict)
        reduced_values.append((_hero_name, results))

    return_dict = {}

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 3
    color = (255, 255, 0)

    thickness = 2
    fontScale = 0.5 * (rows.get_avg_width()/100)

    hero_count = 0
    try:
        GV.THREADS["IMAGE_DB"].join()
    except KeyError:
        pass
    for _hero_name, _hero_data in reduced_values:
        name = _hero_data["pseudo_name"]
        hero_info, _ = GV.IMAGE_DB.search(heroesDict[_hero_name]["image"])
        _hero_data["result"].insert(0, hero_info.name)
        if faction:
            _hero_data["result"].append(hero_info.faction)

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
            if hero_data[0].lower() != "food":
                if debug_raw:
                    _raw_score = return_dict[_row_item.name]["score"]
                    hero_data.append(_raw_score)
                temp_list.append(hero_data)
        json_dict[image_name]["heroes"].append(temp_list)

    return json_dict


def detect_features(name, image_info):
    """
    """
    try:
        GV.THREADS["MODEL"].join()
    except KeyError:
        pass

    return_dict = {}

    test_img = image_info["image"]
    test_img = test_img[..., ::-1]
    results = GV.MODEL([test_img], size=416)

    results_array = results.pandas().xyxy[0]
    RA = results_array
    fi_filtered_results = RA.loc[RA['class'].isin(FI_LABELS.keys())]

    star_filtered_results = RA.loc[RA['class'].isin(
        ASCENSION_STAR_LABELS.keys())]

    si_filtered_results = RA.loc[RA['class'].isin(SI_LABELS.keys())]

    if len(fi_filtered_results) > 0:
        fi_final_results = fi_filtered_results.sort_values(
            "confidence").iloc[0]
        best_fi = FI_LABELS[fi_final_results["class"]]

        fi_scores = {best_fi:
                     fi_final_results["confidence"]}
        if fi_final_results["confidence"] < 0.85:
            best_fi = "0"
            fi_scores = {best_fi: 1.0}
    else:
        best_fi = "0"
        fi_scores = {best_fi: 1.0}
    star = False
    if len(star_filtered_results) > 0:
        final_star_results = star_filtered_results.sort_values(
            "confidence", ascending=False).iloc[0]
        best_ascension = ASCENSION_STAR_LABELS[final_star_results["class"]]

        ascension_scores = {best_ascension:
                            final_star_results["confidence"]}
        if final_star_results["confidence"] > 0.75:
            star = True

    if len(si_filtered_results) > 0:
        final_si_results = si_filtered_results.sort_values(
            "confidence", ascending=False).iloc[0]
        best_si = SI_LABELS[final_si_results["class"]]

        si_scores = {best_si:
                     final_si_results["confidence"]}
        if final_si_results["confidence"] < 0.85:
            best_si = "0"
            si_scores = {best_si: 1.0}
    else:
        best_si = "0"
        si_scores = {best_si: 1.0}
    if not star:
        try:
            GV.THREADS["BORDER_MODEL"].join()
        except KeyError:
            pass

        raw_border_results = GV.BORDER_MODEL(test_img)
        border_results = raw_border_results["instances"]

        classes = border_results.pred_classes.cpu().tolist()

        scores = border_results.scores.cpu().tolist()
        class_list = list(zip([BORDER_MODEL_LABELS[class_num]
                          for class_num in classes], scores))
        if len(class_list) > 0:
            best_class = class_list[0]
            best_ascension = best_class[0]
            ascension_scores = {best_ascension: best_class[1]}
        else:
            best_ascension = "E"
            ascension_scores = {best_ascension: 1.0}

    coords = (image_info["object"].dimensions.x,
              image_info["object"].dimensions.y)
    detection_results = [best_si, best_fi, best_ascension]

    return_dict["score"] = {}
    return_dict["score"]["si"] = si_scores
    return_dict["score"]["fi"] = fi_scores
    return_dict["score"]["ascension"] = ascension_scores

    return_dict["result"] = detection_results
    return_dict["pseudo_name"] = name
    return_dict["coords"] = coords
    return return_dict


if __name__ == "__main__":
    start_time = time.time()
    json_dict = get_si(GV.image_ss, GV.IMAGE_SS_NAME,
                       faction=False)
    if GV.VERBOSE_LEVEL >= 1:
        end_time = time.time()
        print(f"Detected features in: {end_time - start_time}")
        load.display_image(GV.image_ss, display=True)
    if GV.VERBOSE_LEVEL == 0:
        print(f"{{\"heroes\": {json_dict[GV.IMAGE_SS_NAME]['heroes']}}}")
    else:
        print("Heroes:")
        print(f"Rows: {json_dict[GV.IMAGE_SS_NAME]['rows']}")
        print(f"Columns: {json_dict[GV.IMAGE_SS_NAME]['columns']}")
        indent_level = 0
        hero_count = 0
        for row_index, row in enumerate(json_dict[GV.IMAGE_SS_NAME]['heroes']):
            tab_string = "\t" * indent_level
            print(f"{tab_string}Row {row_index + 1}")
            indent_level += 1
            tab_string = "\t" * indent_level
            for hero_index, hero_info in enumerate(row):
                print(f"{tab_string}Hero: {hero_count + 1} {tab_string} "
                      f"{hero_info}")
                hero_count += 1
            indent_level -= 1
