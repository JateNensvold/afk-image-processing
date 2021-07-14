import cv2
import os

import image_processing.load_images as load
import image_processing.globals as GV
import dill
import pickle
import time
import image_processing.database.imageDB as imageSearchDB


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
        if os.path.isdir(_file_path) and _file_path[0] != "_" and "other" not \
                in _file_path:
            recurse_dir(_file_path, file_dict)
        elif os.path.isfile(_file_path) and not _file_path.endswith(".py"):
            if folder not in file_dict:
                file_dict[folder] = set()
            file_dict[folder].add(_file_path)


def buildDB(enrichedDB: bool = False) -> imageSearchDB.imageSearch:
    """
    Build and save a new hero database
    Args:
        enrichedDB: flag to add every hero to the database a second time with
            parts of the the image border removed from each side
    Return:
        imageSearchDB.imageSearch database object
    """
    file_dict = {}
    if GV.VERBOSE_LEVEL >= 1:
        print("Building database!")
    start_time = time.time()
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

    imageDB: imageSearchDB.imageSearch = load.build_flann(baseImages)

    if enrichedDB:
        croppedImages = load.crop_heroes(
            [i[1] for i in baseImages], 0.15, 0.08, 0.25, 0.2)
        for index, cropIMG in enumerate(croppedImages):
            # load.display_image(cropIMG, display=True)
            imageDB.add_image(baseImages[index][0], cropIMG)
        imageDB.matcher.train()
    end_time = time.time()
    with open(GV.database_pickle, 'wb') as handle:
        dill.dump(imageDB, handle)
    if GV.VERBOSE_LEVEL >= 1:
        print("Database built! Built in {} seconds".format(
            end_time-start_time))
    return imageDB


def loadDB() -> imageSearchDB.imageSearch:
    """
    Load hero database from pickle file.

    Return:
        imageSearchDB.imageSearch database object
    """
    if os.path.exists(GV.database_pickle):
        if GV.VERBOSE_LEVEL >= 1:
            print("Loading database!")
        start_time = time.time()
        with open(GV.database_pickle, 'rb') as handle:
            db: imageSearchDB.imageSearch = pickle.load(handle)
        end_time = time.time()
        if GV.VERBOSE_LEVEL >= 1:
            print("Database loaded! Loaded in {} seconds".format(
                end_time - start_time))
        db.matcher.train()

    else:
        raise FileNotFoundError(
            "Unable to find {}. Please call image_processing.build_db.buildDB "
            "to generate a new database".format(GV.database_pickle))
    return db


def get_db(rebuild=GV.REBUILD, enrichedDB=True):
    if rebuild:
        DB = buildDB(enrichedDB=enrichedDB)
    else:
        try:
            DB = loadDB()
        except FileNotFoundError:
            DB = buildDB(enrichedDB=enrichedDB)
    return DB


if __name__ == "__main__":
    get_db()
