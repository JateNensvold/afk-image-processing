# import some common libraries
import os
import cv2
import glob

import image_processing.globals as GV

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

from detectron2.utils.logger import setup_logger
setup_logger()

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def set_config(cfg: CfgNode, train_name: str, test_name: str,
               num_classes: int):

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (test_name,)

    cfg.DATALOADER.NUM_WORKERS = 0
    # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = ("detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/"
                         "137849458/model_final_280758.pkl")

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025

    # adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.MAX_ITER = 3000

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes


def train_model(cfg: CfgNode):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def test_model(cfg: CfgNode):

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(os.path.join(
        GV.TESTS_DIR, "data", "images", "image11.png"))
    output = predictor(image)
    print(output)


def find_json(folder_path: str):
    full_path = os.path.join(folder_path, "*.json")
    glob_list = glob.glob(full_path)
    assert len(glob_list) > 0, f"No Json Found at {folder_path}"
    return glob_list[0]


if __name__ == "__main__":
    base_path = os.path.join(BASE_PATH, "coco_data")
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "valid")
    test_path = os.path.join(base_path, "test")

    register_coco_instances("border_dataset_train", {},
                            find_json(train_path), train_path)
    register_coco_instances("border_dataset_val", {},
                            os.path.join(val_path, "coco.json"), val_path)
    register_coco_instances("border_dataset_test", {},
                            os.path.join(test_path, "coco.json"), test_path)

    cfg = get_cfg()
    set_config(cfg, "border_dataset_train", "border_dataset_val", 12)
    train_model(cfg)
    # test_model(cfg)
