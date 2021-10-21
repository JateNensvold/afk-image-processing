import cv2
import os

import image_processing.load_images as load
import image_processing.globals as GV
import dill
import pickle
import time
import image_processing.database.imageDB as imageSearchDB
import image_processing.afk.hero_object as hero_object


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
    """
    Recursively iterate through directory adding all ".jpg" and ".png" files
        to file_dict
    Args:
        path: folder path to recurse through
        file_dict: file dictionary to add images to
    Return:
        None, updates file_dict passed in
    """
    hero_name = os.path.basename(path)
    faction = os.path.basename(os.path.dirname(path))

    for _file in os.listdir(path):
        _file_path = os.path.join(path, _file)
        # Skip directories with "other" in them or that start with "_"
        if os.path.isdir(_file_path) and _file_path[0] != "_" and "other" not \
                in _file_path:
            recurse_dir(_file_path, file_dict)
        elif os.path.isfile(_file_path) and (_file_path.endswith(".png") or
                                             _file_path.endswith(".jpg")):
            if faction not in file_dict:
                file_dict[faction] = {}
            if hero_name not in file_dict[faction]:
                file_dict[faction][hero_name] = set()
            file_dict[faction][hero_name].add(_file_path)


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
    base_images = []

    for _faction, _faction_heroes in file_dict.items():
        for _hero_name, _hero_paths in _faction_heroes.items():
            for _hero_path in _hero_paths:
                if not os.path.exists(_hero_path):
                    raise FileNotFoundError(_hero_path)
                hero = cv2.imread(_hero_path)
                if hero is None:
                    raise FileNotFoundError(
                        "Hero not found: {}".format(_hero_path))

                base_images.append(hero_object.hero_object(
                    _hero_name, _faction, hero))

    imageDB: imageSearchDB.imageSearch = load.build_flann(base_images)

    if enrichedDB:
        croppedImages = load.crop_heroes(
            [_hero_tuple.image for _hero_tuple in base_images],
            0.15, 0.08, 0.25, 0.2)
        for index, cropIMG in enumerate(croppedImages):
            # load.display_image(cropIMG, display=True)
            imageDB.add_image(base_images[index], cropIMG)
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
    GV.IMAGE_DB = DB
    return DB


if __name__ == "__main__":
    get_db()
