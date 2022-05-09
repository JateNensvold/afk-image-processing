import os
from pathlib import Path
import re
import time
import pickle
import typing
from typing import List, Set, Union, Dict

import cv2
import dill

import image_processing.globals as GV
from image_processing.afk.hero.hero_data import HeroImage
from image_processing.database.image_database import build_flann


if typing.TYPE_CHECKING:
    from image_processing.database.image_database import ImageSearch


def find_images(folder_path: str, file_dict: Dict[str, Set]):
    """_summary_
    Recursively iterate through directory adding all ".jpg" and ".png" files
        to file_dict
    Args:
        folder_path (str): folder path to recurse through
        file_dict (dict): file dictionary to add images to

    """
    for hero_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, hero_name)
        if os.path.isfile(file_path) and (file_path.endswith(".png") or
                                          file_path.endswith(".jpg") or
                                          file_path.endswith(".webp")):
            if hero_name not in file_dict:
                file_dict[hero_name] = set()
            file_dict[hero_name].add(file_path)


def build_database(enriched_db: bool = False) -> "ImageSearch":
    """
    Build and save a new hero database
    Args:
        enriched_db: flag to add every hero to the database a second time with
            parts of the the image border removed from each side
    Return:
        "ImageSearch" database object
    """
    file_dict: Dict[str, Union[Dict, Set]] = {}
    if GV.verbosity(1):
        print("Building database!")
    start_time = time.time()
    for hero_portrait_dir in GV.HERO_PORTRAIT_DIRECTORIES:
        find_images(hero_portrait_dir, file_dict)
    base_images: List[HeroImage] = []

    for raw_hero_name, hero_path_set in file_dict.items():
        for hero_path in hero_path_set:
            if not os.path.exists(hero_path):
                raise FileNotFoundError(hero_path)
            hero_image = cv2.imread(hero_path)
            if hero_image is None:
                raise FileNotFoundError(
                    f"Hero Image not found: {hero_path}")
            hero_name, *_ = re.split(r"\.", raw_hero_name)
            base_images.append(HeroImage(
                hero_name, hero_image, hero_path))
    image_db: "ImageSearch" = build_flann(base_images, enriched_db=enriched_db)

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
    return image_db


if __name__ == "__main__":
    GV.VERBOSE_LEVEL = 2
    # GV.DEBUG = True
    get_db(rebuild=True, enriched_db=True)
