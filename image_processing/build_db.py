import cv2
import os
import pickle

import image_processing.load_images as load
import image_processing.globals as GV
import dill


def pickle_trick(obj, max_depth=10):
    output = {}

    if max_depth <= 0:
        return output

    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError) as e:
        failing_children = []

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)

        output = {
            "fail": obj,
            "err": e,
            "depth": max_depth,
            "failing_children": failing_children
        }

    return output


def recurse_dir(path: str, file_dict: dict):
    folder = os.path.basename(os.path.basename(path))
    for _file in os.listdir(path):
        _file_path = os.path.join(path, _file)
        if os.path.isdir(_file_path) and _file_path[0] != "_" and "other" not in _file_path:
            recurse_dir(_file_path, file_dict)
        elif os.path.isfile(_file_path) and not _file_path.endswith(".py"):
            if folder not in file_dict:
                file_dict[folder] = set()
            file_dict[folder].add(_file_path)


def buildDB(enrichedDB: bool = False):
    """
    Build and save a new hero database
    Args:
        enrichedDB: flag to add every hero to the database a second time with
            0.15 of their image removed one each side
    Return:
        a load.imageSearch() object filled with heroes
    """
    file_dict = {}
    print("Building database!")

    recurse_dir(GV.database_icon_path, file_dict)
    baseImages = []
    for _hero_name, _hero_paths in file_dict.items():
        if _hero_name == "hero_icon":
            print(_hero_paths)
        for _hero_path in _hero_paths:
            if not os.path.exists(_hero_path):
                print(_hero_path)
                raise FileNotFoundError()
            hero = cv2.imread(_hero_path)
            # name = os.path.basename(_hero_path)
            baseImages.append((_hero_name, hero))

    imageDB = load.build_flann(baseImages)

    if enrichedDB:
        croppedImages = load.crop_heroes(
            [i[1] for i in baseImages], 0.15, 0.08, 0.25, 0.2)
        for index, cropIMG in enumerate(croppedImages):
            # load.display_image(cropIMG, display=True)
            imageDB.add_image(baseImages[index][0], cropIMG)
        imageDB.matcher.train()

    with open('imageDB.pickle', 'wb') as handle:
        dill.dump(imageDB, handle)
    # print(len(imageDB.names))
    print("Database built!")
    return imageDB


def loadDB(Refresh: bool):
    """
    Load/Refresh hero database

    Return:
        load.
    """
    db = None
    with open('imageDB.pickle', 'rb') as handle:
        db = pickle.load(handle)
    print(dir(db))
    return db


if __name__ == "__main__":
    buildDB()
