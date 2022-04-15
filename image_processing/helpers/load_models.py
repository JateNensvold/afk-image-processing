import threading
import time
import torch

import image_processing.helpers.verbose_print as VP
import image_processing.build_db as BD
import image_processing.globals as GV
import image_processing.helpers.load_models as LM


def load_files(model_path: str, enriched_db=True,
               thread_wait: bool = True):
    """
    Load the pytorch and detectron models and hero image database

    Args:
        model_path (str): path to FI/SI/Stars pytorch model
        enriched_db (bool, optional): flag to add every hero to the database a
            second time with parts of the the image border removed from each
            side. Defaults to True.
        thread_wait (bool, optional): flag that causes program to wait while
            the threads loading the models and database execute. Defaults to
            True.
    """
    if GV.IMAGE_DB is None:
        db_thread = threading.Thread(
            kwargs={"enriched_db": enriched_db},
            target=BD.get_db)
        GV.THREADS["IMAGE_DB"] = db_thread
        db_thread.start()

    if GV.FI_SI_STAR_MODEL is None:
        model_thread = threading.Thread(
            args=[model_path],
            target=LM.load_fi_model)
        GV.THREADS["FI_SI_STAR_MODEL"] = model_thread
        model_thread.start()

    if thread_wait:
        for _thread_name, thread in GV.THREADS.items():
            thread.join()


def load_fi_model(model_path: str):
    """
    Load hero FI/SI/Stars model into GV.FI_SI_STAR_MODEL

    Args:
        model_path (str): path to model being loaded
    """
    start_time = time.time()

    fi_si_model = torch.hub.load(
        str(GV.YOLOV5_DIR),
        "custom",
        model_path,
        source="local")

    end_time = time.time()
    VP.print_verbose(
        f"Loaded FI/SI/Ascension model({model_path}) in: {end_time - start_time}", verbose_level=1)
    GV.FI_SI_STAR_MODEL = fi_si_model
