import os
from pathlib import Path
import time
import pickle
import typing

import cv2
import dill

import image_processing.globals as GV
import image_processing.load_images as load
import image_processing.afk.hero_object as HO
from image_processing.database.image_database import build_flann


if typing.TYPE_CHECKING:
    from image_processing.database.image_database import ImageSearch


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


def build_database(enriched_db: bool = False) -> "ImageSearch":
    """
    Build and save a new hero database
    Args:
        enriched_db: flag to add every hero to the database a second time with
            parts of the the image border removed from each side
    Return:
        "ImageSearch" database object
    """
    file_dict = {}
    if GV.verbosity(1):
        print("Building database!")
    start_time = time.time()
    recurse_dir(GV.HERO_ICON_DIR, file_dict)
    base_images = []

    for _faction, _faction_heroes in file_dict.items():
        for _hero_name, _hero_paths in _faction_heroes.items():
            for hero_path in _hero_paths:
                if not os.path.exists(hero_path):
                    raise FileNotFoundError(hero_path)
                hero = cv2.imread(hero_path)
                if hero is None:
                    raise FileNotFoundError(
                        f"Hero not found: {hero_path}")

                base_images.append(HO.hero_object(
                    _hero_name, _faction, hero))

    image_db: "ImageSearch" = build_flann(base_images)

    if enriched_db:
        cropped_images = load.crop_heroes(
            [_hero_tuple.image for _hero_tuple in base_images],
            0.15, 0.08, 0.25, 0.2)
        for index, cropped_image in enumerate(cropped_images):
            # load.display_image(cropIMG, display=True)
            image_db.add_image(base_images[index], cropped_image)
        image_db.matcher.train()
    end_time = time.time()
    with open(GV.DATABASE_PICKLE_PATH, 'wb') as handle:
        dill.dump(image_db, handle)
    if GV.verbosity(1):
        print(f"Database built! Built in {end_time-start_time} seconds")
    return image_db


def load_database(
        pickle_path: Path = GV.DATABASE_PICKLE_PATH) -> "ImageSearch":
    """
    Load hero database from pickle file.

    Return:
        "ImageSearch" database object
    """
    if pickle_path.exists():
        if GV.verbosity(1):
            print("Loading database!")
        start_time = time.time()
        with open(pickle_path, 'rb') as handle:
            image_db: "ImageSearch" = pickle.load(handle)
        end_time = time.time()
        if GV.verbosity(1):
            print(
                f"Database loaded! Loaded in {end_time - start_time} seconds")
        image_db.matcher.train()

    else:
        raise FileNotFoundError(
            f"Unable to find {GV.DATABASE_PICKLE_PATH}. Please call "
            "image_processing.build_db.build_database to generate a new "
            "database")
    return image_db


def get_db(rebuild: bool = GV.REBUILD, enriched_db=True):
    """
    Try to fetch 'ImageSearch' database from disk, rebuild the database if
        the fetch fails

    Args:
        rebuild ([bool], optional): Flag to rebuild the database when its True.
             Defaults to GV.REBUILD.
        enriched_db (bool, optional): flag to add every hero to the database a
            second time with parts of the the image border removed from each
            side, only valid when database is being built/rebuild.
            Defaults to True.

    Returns:
        [type]: [description]
    """
    if rebuild:
        image_db = build_database(enriched_db=enriched_db)
    else:
        try:
            image_db = load_database()
        except FileNotFoundError:
            image_db = build_database(enriched_db=enriched_db)
    GV.IMAGE_DB = image_db
    return image_db


if __name__ == "__main__":
    GV.VERBOSE_LEVEL = 1
    get_db(rebuild=True, enriched_db=True)
