import time
import warnings
from multiprocessing.pool import ThreadPool

import torch

import image_processing.utils.verbose_print as VP
import image_processing.build_db as BD
import image_processing.globals as GV


def load_files(attributes_model_path: str, ascension_model_path: str,
               enriched_db=True):
    """
    Load the pytorch and detectron models and hero image database

    Args:
        model_path (str): path to FI/SI/Stars pytorch model
        enriched_db (bool, optional): flag to add every hero to the database a
            second time with parts of the the image border removed from each
            side. Defaults to True.
    """
    # Silence Module loading warnings from pytorch
    warnings.filterwarnings("ignore")

    thread_pool = ThreadPool(processes=3)

    attribute_model_thread = thread_pool.apply_async(
        load_model, [attributes_model_path])
    ascension_model_thread = thread_pool.apply_async(
        load_model, [ascension_model_path])
    hero_database_thread = thread_pool.apply_async(
        BD.get_db, [], {"enriched_db": enriched_db})

    GV.IMAGE_DB = hero_database_thread.get()
    GV.ASCENSION_BORDER_MODEL = ascension_model_thread.get()
    GV.FI_SI_STAR_MODEL = attribute_model_thread.get()


def load_model(model_path: str):
    """
    Load hero FI/SI/Stars model into GV.FI_SI_STAR_MODEL

    Args:
        model_path (str): path to model being loaded
    """
    start_time = time.time()

    yolov5_model = torch.hub.load(
        str(GV.YOLOV5_DIR),
        "custom",
        model_path,
        source="local")

    end_time = time.time()
    VP.print_verbose(
        f"Loaded model({model_path}) in: {end_time - start_time}", verbose_level=1)

    return yolov5_model
