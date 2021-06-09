

import csv
import json
import datetime
import os
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer

INFO = {
    "description": "AFK Arena Hero Info Dataset",
    "url": "https://github.com/JateNensvold/afk-image-processing",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "jatenensvold",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = {
    "none": {
        'id': 0,
        'name': 'none_si',
        'supercategory': 'si',
    },
    "0": {
        'id': 1,
        'name': '0_si',
        'supercategory': 'si',
    },
    "10": {
        'id': 2,
        'name': '10_si',
        'supercategory': 'si',
    },
    "20": {
        'id': 3,
        'name': '20_si',
        'supercategory': 'si',
    },
    "30": {
        'id': 4,
        'name': '30_si',
        'supercategory': 'si',
    },
}


def get_dataset_function():

    with open("si_data.json", "r") as fp:
        data = json.load(fp)
    dataset = []
    for index, image_data in enumerate(data["images"]):
        record = {**image_data}
        record["annotations"] = []
        anno = data["annotations"][index]
        tempAnno = {}
        tempAnno["bbox"] = anno["bbox"]
        tempAnno["bbox_mode"] = anno["bbox_mode"]
        tempAnno["category_id"] = anno["category_id"]
        record["annotations"].append(tempAnno)
        dataset.append(record)
    return dataset


def init():
    csvfile = open("./si_data.txt", "r")
    jsonfile = open("./si_data.json", "w")

    fieldNames = ["path", "left", "bottom", "right", "top", "label"]
    reader = csv.DictReader(csvfile, fieldNames)
    id = 0
    data_list = []
    annotation_list = []
    for row in reader:
        left = int(row["left"])
        right = int(row["right"])
        top = int(row["top"])
        bottom = int(row["bottom"])

        try:
            path = row["path"]
            image = cv2.imread(path)
            width, height = image.shape[:2]
            folder = os.path.basename(os.path.dirname(path))
        except Exception:
            dirname, basename = os.path.split(path)
            dirname, folder = os.path.split(dirname)
            folder = "none"
            path = os.path.join(dirname, folder, basename)
        finally:
            print(path)
            image = cv2.imread(path)

            try:
                height, width = image.shape[:2]
            except Exception:
                continue

        data = {}
        data["id"] = id
        data["file_name"] = path
        # data["file_name"] = os.path.basename(path)

        data["path"] = path

        height, width = image.shape[:2]
        data["width"] = width
        data["height"] = height
        timestamp = os.path.getctime(path)
        formatedDate = datetime.datetime.utcfromtimestamp(
            timestamp).strftime("%Y-%m-%d %H:%M:%S")
        data["date_captured"] = formatedDate
        data["license"] = 1
        data["coco_url"] = ""
        data["flickr_url"] = ""

        data_list.append(data)

        annotation = {}
        annotation["id"] = id
        annotation["image_id"] = id

        annotation["category_id"] = CATEGORIES[folder.lower()]["id"]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS
        annotation["iscrowd"] = 0
        annotation["area"] = (int(bottom)-int(top)) * \
            (int(right)-int(left))
        # left, top, width, height
        annotation["bbox"] = [left,
                              top,
                              right-left,
                              bottom-top]
        annotation["width"] = width
        annotation["height"] = height

        annotation_list.append(annotation)
        id += 1

    json_dict = {"info": INFO,
                 "licenses": LICENSES,
                 "categories": [v for k, v in CATEGORIES.items()],
                 "images": data_list,
                 "annotations": annotation_list}
    with open("si_data.json", "w") as fp:
        json.dump(json_dict, fp)
    DatasetCatalog.register("AFK_arena_SI_info",
                            get_dataset_function)
    classNames = [v["name"] for k, v in CATEGORIES.items()]
    print(classNames)
    # MetadataCatalog.get("AFK_arena_SI_info").set(
    #     thing_classes=classNames)
    SI_metadata = MetadataCatalog.get("AFK_arena_SI_info")
    classes = [k for k, v in CATEGORIES.items()]
    print(classes)
    input()
    SI_metadata.thing_classes = classes

    train()


def train():
    import torch
    torch.autograd.set_detect_anomaly(True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("AFK_arena_SI_info",)
    cfg.DATASETS.TEST = ()
    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    # 300 iterations seems good enough for this toy dataset; you will need to
    #   train longer for a practical dataset
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = (1000, 1499)        # do not decay learning rate
    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    # only has one class (ballon). (see https://detectron2.readthedocs.io/
    #   tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    # NOTE: this config means the number of classes, but a few popular
    #   unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()


def classify():
    cfg = get_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    SI_metadata = MetadataCatalog.get("AFK_arena_SI_info")
    import matplotlib.pyplot as plt
    directory = "./backup/30"
    for imagePath in sorted(os.listdir(directory)):
        im = cv2.imread(os.path.join(directory, imagePath))
        outputs = predictor(im)
        print(type(outputs["instances"]))
        print(outputs)
        v = Visualizer(im[:, :, ::-1],
                       metadata=SI_metadata,
                       scale=0.5)
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_image = result.get_image()[:, :, ::-1]
        # plt.figure()
        # plt.imshow(result_image)
        # plt.show()
        cv2.imshow(imagePath, result_image)
        print(dir(outputs["instances"]))
        input("Press any key to continue... ")
        # plt.close()


if __name__ == "__main__":

    # init()
    classify()
