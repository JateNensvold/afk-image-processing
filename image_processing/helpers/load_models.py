import threading
import time
import torch

import image_processing.helpers.verbose_print as VP
import image_processing.build_db as BD
import image_processing.globals as GV
import image_processing.helpers.load_models as LM
import image_processing.models.model_attributes as MA

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


def load_files(model_path: str, border_model_path: str, enriched_db=True,
               thread_wait: bool = True):
    """
    Load the pytorch and detectron models and hero image database

    Args:
        model_path (str): path to FI/SI/Stars pytorch model
        border_model_path (str): path to Ascension/Border detectron2 model
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

    if GV.MODEL is None:
        model_thread = threading.Thread(
            args=[model_path],
            target=LM.load_fi_model)
        GV.THREADS["MODEL"] = model_thread
        model_thread.start()

    if GV.BORDER_MODEL is None:
        border_model_thread = threading.Thread(
            args=[border_model_path, MA.BORDER_MODEL_LABELS],
            target=LM.load_border_model)
        GV.THREADS["BORDER_MODEL"] = border_model_thread
        border_model_thread.start()

    if thread_wait:
        for _thread_name, thread in GV.THREADS.items():
            thread.join()


def load_border_model(model_path: str, border_labels: list):
    """
    Load Hero Ascension/Border detectron2 model

    Args:
        path (str): path to model being loaded
        border_labels (list): list of classes/labels in model
    """
    start_time = time.time()
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.DEVICE = GV.ARCHITECTURE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(border_labels)
    cfg.MODEL.WEIGHTS = model_path

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    border_model = DefaultPredictor(cfg)
    end_time = time.time()
    VP.print_verbose(f"Loaded Border_model in: {end_time - start_time}",
                     verbose_level=1)
    GV.BORDER_MODEL = border_model


def load_fi_model(model_path: str):
    """
    Load hero FI/SI/Stars model into GV.MODEL

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
    GV.MODEL = fi_si_model
