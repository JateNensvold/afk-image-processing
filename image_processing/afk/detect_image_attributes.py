"""
Module that wraps the top level API for the AFK Arena Roster Screenshot and
Hero processing

Calling detect_features assumes that the image processing environment has been
initialized and will parse apart an image and feed it into the various models
needed to detect AFK Arena Hero Features
"""
from typing import Tuple, Dict
import warnings

import cv2
import numpy as np

import image_processing.globals as GV
import image_processing.processing as processing
import image_processing.models.model_attributes as MA
import image_processing.afk.roster.matrix as matrix

warnings.filterwarnings("ignore")
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (255, 255, 0)

THICKNESS = 2


def detect_features(roster_image: np.ndarray, debug_raw: bool = None,
                    detect_faction: bool = False):
    """
    Detect AFK Arena heroes from a roster screenshot and for each hero detect
        "FI", "SI", "Ascension", and "hero Name"
    Args:
        roster_image: image to run segmentation and detection on

        debug_raw: flag to add raw values for SI, FI and Ascension detection
            to return dictionary
        detect_faction: flag to add faction output to hero feature list in the
            return dict
    """

    if debug_raw is None:
        if GV.verbosity(1):
            debug_raw = True
        else:
            debug_raw = False

    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([179, 255, 192])

    hsv_range = [lower_hsv, upper_hsv]
    blur_args = {"hsv_range": hsv_range}
    heroes_dict, hero_matrix = processing.get_heroes(
        roster_image, blur_args)

    reduced_values: list[Tuple[str, Dict]] = []
    for pseudo_name, image_info in heroes_dict.items():
        # detect features with models
        results = detect_attributes(
            pseudo_name, image_info)
        reduced_values.append((pseudo_name, results))

    return_dict = {}

    hero_count = 0

    for pseudo_name, hero_data in reduced_values:
        hero_info, _template_image = GV.IMAGE_DB.search(
            heroes_dict[pseudo_name]["image"])
        # Add detected hero name to start of result list
        hero_data["result"].insert(0, hero_info.name)

        # Add faction to end of result list if
        if detect_faction:
            hero_data["result"].append(hero_info.faction)

        return_dict[pseudo_name] = hero_data
        if GV.verbosity(1):
            label_hero_feature(roster_image, hero_data, hero_matrix)

        hero_count += 1

    json_dict = {}

    json_dict["rows"] = len(hero_matrix)
    json_dict["columns"] = max([len(_row) for _row in hero_matrix])

    json_dict["heroes"] = []

    for _row in hero_matrix:
        temp_list = []
        for _row_item in _row:
            hero_data = return_dict[_row_item.name]["result"]
            if hero_data[0].lower() != "food":
                if debug_raw:
                    _raw_score = return_dict[_row_item.name]["score"]
                    hero_data.append(_raw_score)
                temp_list.append(hero_data)
        json_dict["heroes"].append(temp_list)

    return json_dict


def label_hero_feature(roster_image: np.ndarray,
                       hero_data: processing.HERO_INFO,
                       hero_matrix: matrix.Matrix):
    """
    Write hero data such as FI/SI/Stars onto the roster image those attributes
        were derived from

    Args:
        roster_image (np.ndarray): roster/image of heroes
        hero_data (processing.HERO_INFO): data about a hero on the
            'roster_image'
        hero_matrix (matrix.matrix): matrix of hero data in the same horizontal
            and vertical order they were detected in
    """

    font_scale = 0.5 * (hero_matrix.get_avg_width()/100)

    result = str(hero_data["result"])
    coords = hero_data["coords"]
    text_size = cv2.getTextSize(result, FONT, font_scale, THICKNESS)
    height = text_size[0][1]
    coords = (coords[0], coords[1] + round(5 * height))

    cv2.putText(
        roster_image, result, coords, FONT, abs(
            font_scale), COLOR, THICKNESS,
        cv2.LINE_AA)


def detect_attributes(pseudo_name: str, image_info: processing.HERO_INFO):
    """
    Detect hero features such as FI, SI, Stars and ascension level using'
        custom trained yolov5 and detectron2 image recognition models

    Args:
        pseudo_name (str): name generated for hero during matrix detection
        image_info (processing.HERO_INFO): dictionary of hero information
            including image to be passed to model
    Returns:
        [type]: [description]
    """

    return_dict = {}

    test_img = image_info["image"]
    test_img = test_img[..., ::-1]
    results = GV.MODEL([test_img], size=416)  # pylint: disable=not-callable

    results_array = results.pandas().xyxy[0]
    fi_filtered_results = results_array.loc[results_array['class'].isin(
        MA.FI_LABELS.keys())]

    star_filtered_results = results_array.loc[results_array['class'].isin(
        MA.ASCENSION_STAR_LABELS.keys())]

    si_filtered_results = results_array.loc[results_array['class'].isin(
        MA.SI_LABELS.keys())]

    if len(fi_filtered_results) > 0:
        fi_final_results = fi_filtered_results.sort_values(
            "confidence").iloc[0]
        best_fi = MA.FI_LABELS[fi_final_results["class"]]

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
        best_ascension = MA.ASCENSION_STAR_LABELS[final_star_results["class"]]

        ascension_scores = {best_ascension:
                            final_star_results["confidence"]}
        if final_star_results["confidence"] > 0.75:
            star = True

    if len(si_filtered_results) > 0:
        final_si_results = si_filtered_results.sort_values(
            "confidence", ascending=False).iloc[0]
        best_si = MA.SI_LABELS[final_si_results["class"]]

        si_scores = {best_si:
                     final_si_results["confidence"]}
        if final_si_results["confidence"] < 0.85:
            best_si = "0"
            si_scores = {best_si: 1.0}
    else:
        best_si = "0"
        si_scores = {best_si: 1.0}
    if not star:
        # pylint: disable=not-callable
        raw_border_results = GV.BORDER_MODEL(
            test_img)
        border_results = raw_border_results["instances"]

        classes = border_results.pred_classes.cpu().tolist()

        scores = border_results.scores.cpu().tolist()
        class_list = list(zip([MA.BORDER_MODEL_LABELS[class_num]
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
    if GV.verbosity(1):
        return_dict["pseudo_name"] = pseudo_name
    return_dict["coords"] = coords
    return return_dict


# if __name__ == "__main__":
#     start_time = time.time()
#     json_dict = get_si(GV.IMAGE_SS, detect_faction=False)
#     if GV.verbosity(1):
#         end_time = time.time()
        # print(f"Detected features in: {end_time - start_time}")
#         load.display_image(GV.IMAGE_SS, display=True)
#     if GV.VERBOSE_LEVEL == 0:
#         print(f"{{\"heroes\": {json_dict[GV.IMAGE_SS_NAME]['heroes']}}}")
#     else:
#         print("Heroes:")
#         print(f"Rows: {json_dict[GV.IMAGE_SS_NAME]['rows']}")
#         print(f"Columns: {json_dict[GV.IMAGE_SS_NAME]['columns']}")
#         indent_level = 0
#         hero_count = 0
#         for row_index, row in enumerate(json_dict[GV.IMAGE_SS_NAME]['heroes']):
#             tab_string = "\t" * indent_level
#             print(f"{tab_string}Row {row_index + 1}")
#             indent_level += 1
#             tab_string = "\t" * indent_level
#             for hero_index, hero_info in enumerate(row):
#                 print(f"{tab_string}Hero: {hero_count + 1} {tab_string} "
#                       f"{hero_info}")
#                 hero_count += 1
#             indent_level -= 1
