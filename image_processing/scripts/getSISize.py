import os
import statistics

import cv2
import numpy as np

from image_processing.processing.image_processing import blur_image
from image_processing.stamina import digit_features
import image_processing.globals as GV


import imutils
# Need this import to use imutils.contours.sort_contours,
#   without it Module raises AttributeError
# pylint: disable=unused-import
from imutils import contours  # noqa


def get_digit(image):
    """
    Detect and return 'L', 'v', 'l' from image

    Args:
        image: np.array of rgb hero image

    Return:
        list of dictionaries with information about each  'L', 'v', 'l'
        detected
    """

    height, width = image.shape[:2]

    cropped = image[0:round(height*0.3), round(width*0.25):width]

    # (hMin = 11 , sMin = 4, vMin = 110), (hMax = 38 , sMax = 191, vMax = 255)
    lower_bound = np.array([11, 4, 167])
    upper_bound = np.array([38, 191, 255])

    cropped_blur = blur_image(
        cropped, hsv_range=[lower_bound, upper_bound])

    digit_countour = cv2.findContours(
        cropped_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_countour = imutils.grab_contours(digit_countour)
    digit_countour = imutils.contours.sort_contours(digit_countour,
                                                    method="left-to-right")[0]
    bins = {}
    all_y = []

    for index, digit in enumerate(digit_countour):
        x_coord, y_coord, width, height = cv2.boundingRect(digit)
        # if h/w < 10/3:
        # if w > 3 and h > 10:
        bins[index] = {}
        bins[index]["cnt"] = digit
        all_y.append((y_coord, y_coord+height, index))
    top_med = statistics.median([num[0] for num in all_y])
    bottom_med = statistics.median([num[1] for num in all_y])

    final_digits = []

    for digit_tuple in all_y:
        digit_top = digit_tuple[0]
        digit_bottom = digit_tuple[1]
        index = digit_tuple[2]
        if abs(digit_bottom-bottom_med) <= 3 and (digit_top >= (top_med - 3)):
            x_coord, y_coord, width, height = cv2.boundingRect(
                bins[index]["cnt"])
            digit_image = cropped[y_coord:y_coord +
                                  height, x_coord:x_coord+width]
            bins[index]["roi"] = digit_image
            final_digits.append(digit_tuple)
    verified = {}
    # iterate over first 3 "letters" detected
    for digit_tuple in final_digits[:3]:
        index = digit_tuple[2]
        image = bins[index]["roi"]
        bins[index]["digit_info"] = digit_tuple
        results = get_level_digit(cropped, bins[index])
        values_list = list(results.values())
        keys_list = list(results.keys())
        best_label = keys_list[values_list.index(max(values_list))]
        # bestLabel = max(results.items(), key=lambda x: x[0])
        best_match = results[best_label]
        if best_match[0] < 0.9:
            continue
        height = image.shape[1]
        template_height = best_match[2]
        bins[index]["scale"] = height/template_height
        bins[index]["match_info"] = best_match
        # bins[index]["digit_info"] = digitTuple
        verified[best_label] = bins[index]
    return verified


def get_level_digit(source_image: np.ndarray, digit_dict: dict,
                    train: bool = False):
    """[summary]

    Args:
        source_image (np.ndarray): [description]
        digit_dict (dict): [description]
        train (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    folders = ["L", "v", "l"]
    if train:
        train_levels(source_image)

    results = {}
    for folder in folders:
        folder_path = os.path.join(GV.DATABASE_LEVELS_DATA_DIR, folder)
        folder_files = os.listdir(folder_path)
        if folder not in results:
            results[folder] = []
        for file in folder_files:
            file_path = os.path.join(folder_path, file)
            template_image = cv2.imread(file_path)

            height, width = template_image.shape[:2]

            x_coord, y_coord, width, height = cv2.boundingRect(
                digit_dict["cnt"])
            detected_digit = source_image[y_coord:y_coord +
                                          height, x_coord:x_coord+width]

            height_ratio = height/height

            new_height = max(round(height * height_ratio), 1)
            new_width = max(round(width * height_ratio), 1)

            # print(min(newWidth, w), min(new_height, h))
            # print(min(new_height, h)/min(newWidth, w))
            template_resize = cv2.resize(
                template_image, (min(new_width, width),
                                 min(new_height, height)))
            template_match = cv2.matchTemplate(
                detected_digit, template_resize, cv2.TM_CCOEFF_NORMED)
            (_, score, _, _) = cv2.minMaxLoc(template_match)
            results[folder].append((score, file_path, height))
    return_results = {}
    for key, value in results.items():
        return_results[key] = max(value, key=lambda x: x[0])
    return return_results


def train_levels(image):
    """[summary]

    Args:
        image ([type]): [description]
    """
    digit_features(image, save_dir=GV.DATABASE_LEVELS_DATA_DIR)
