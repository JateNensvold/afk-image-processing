import torch
import time

import image_processing.helpers.verbose_print as VP

import image_processing.globals as GV
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


def load_border_model(path: str, border_labels: list):
    """Load
    """

    start_time = time.time()
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("border_dataset_train",)
    cfg.DATASETS.TEST = ("border_dataset_val",)

    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.MODEL.DEVICE = GV.ARCHITECTURE

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025

    # adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.MAX_ITER = 3000

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8

    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(border_labels)
    BORDER_MODEL = DefaultPredictor(cfg)
    end_time = time.time()
    VP.print_verbose("Loaded Border_model in: {}".format(
        end_time - start_time), verbose_level=1)
    GV.BORDER_MODEL = BORDER_MODEL


def load_FI_model(path: str):
    start_time = time.time()
    MODEL = torch.hub.load(
        GV.yolov5_dir,
        "custom",
        path,
        source="local",
        force_reload=True,
        verbose=False)
    end_time = time.time()
    VP.print_verbose(
        "Loaded star/FI model in: {}".format(
            end_time - start_time), verbose_level=1)
    GV.MODEL = MODEL
