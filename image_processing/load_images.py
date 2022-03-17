import glob
from typing import List, NamedTuple, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt

import image_processing.globals as GV


class CropImageInfo(NamedTuple):
    """_summary_
    """
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float


def display_image(image: Union[np.ndarray, List], multiple: bool = False,
                  display: bool = GV.DEBUG, color_correct: bool = True,
                  colormap: bool = False):
    """
    Display the 'image' passed in in the desired manner based on the flags this
        function was called with

    Args:
        image (np.ndarray): image to display
        multiple (bool, optional): [description]. Defaults to False.
        display (bool, optional): [description]. Defaults to GV.DEBUG.
        color_correct (bool, optional): [description]. Defaults to True.
        colormap (bool, optional): [description]. Defaults to False.
    """
    # backend = matplotlib.get_backend()

    # if backend.lower() != 'tkagg':
    #     if GV.verbosity(1):
    #         print("Backend: {}".format(backend))
    #     plt.switch_backend("tkagg")

    if not display:
        return

    if multiple:
        image = concat_resize(image)
    elif color_correct and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.ion()

    plt.figure()
    if colormap:
        plt.imshow(image, cmap="gray")

    else:
        plt.imshow(image)

    plt.show()
    input('Press any key to continue...')
    plt.close("all")


def find_files(path: str, flag=True, lower=False):
    """
    Finds and returns all filepaths that match the pattern/path passed
        in under 'path'

    Args:
        path: glob/regex to path of file(s)
        flag: flag to force file loading if it contains "_" or "-"
        lower: flag to load all filepaths as lowercase
    Return:
        Sorted list of file paths
    """
    valid_images = []
    images = glob.glob(path)
    # images = os.listdir(path)
    for i in images:
        if flag or ("_" not in i and "-"not in i):
            if lower:
                i = i.lower()
            valid_images.append(i)
    return sorted(valid_images)


def crop_heroes(images: list[np.ndarray], crop_info: CropImageInfo,
                border_width=0.25):
    """
    Args:
        images: list of images to crop frame from
        border_width: percentage of border to take off of each side of the
            image
        removeBG: flag to attempt to remove the background from each hero
            returned(ensure this has not already been done to image earlier on
            in the process)
    Returns:
        dict of name as key and  images as values with 'border_width'
            removed from each side of the images
    """
    sides = {"x_left": crop_info.x_left, "x_right": crop_info.x_right,
             "y_top": crop_info.y_top, "y_bottom": crop_info.y_bottom}

    cropped_heroes: List[np.ndarray] = []
    for _name, _side in sides.items():
        if _side is None:
            sides[_name] = border_width

    for image in images:
        shape = image.shape
        x_coord = shape[0]
        y_coord = shape[1]

        left = round(sides["x_left"] * x_coord)
        right = round(sides["x_right"] * x_coord)
        top = round(sides["y_top"] * y_coord)
        bottom = round(sides["y_bottom"] * y_coord)

        crop_img = image[top: x_coord-bottom, left: y_coord-right]
        cropped_heroes.append(crop_img)

    return cropped_heroes


def concat_resize(img_list: List[np.ndarray], interpolation: int = cv2.INTER_CUBIC):
    """
    Concatonate a list of images into a single image, using the tallest images
        height to scale all other images in the list when the concatenation
        occurs

    Args:
        img_list (List): the list of images to concatonate together
        interpolation (int, optional): interpolation method to use.
            Defaults to cv2.INTER_CUBIC.

    Returns:
        [np.ndarray]: single image that is 'img_list' concatenated together
    """

    h_max = max(img.shape[0]
                for img in img_list)
    # resizing images
    im_list_resize = []
    for img in img_list:
        height, width = img.shape[:2]

        scale = h_max/height
        resize: np.ndarray = cv2.resize(img,
                                        (int(width * scale), int(height * scale)),
                                        interpolation=interpolation)

        new_h, new_w = resize.shape[:2]

        canvas = np.zeros((h_max, new_w, 3))
        image_dimensions = len(resize.shape)
        if len(resize.shape) == 2:
            resize = cv2.merge([resize, resize, resize])

        if image_dimensions < 3:
            canvas[0:new_h, 0:new_w] = resize
        else:
            canvas[0:new_h, 0:new_w, 0:3] = resize
        im_list_resize.append(canvas)

    return np.hstack(im_list_resize).astype(np.uint8)
