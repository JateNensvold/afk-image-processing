import os
import argparse
import logging
import threading
import typing

import image_processing.helpers.load_images as load
if typing.TYPE_CHECKING:
    import image_processing.database.imageDB as imageSearchDB

parser = argparse.ArgumentParser(description='AFK arena object extraction and '
                                 'image analysis.')
parser.add_argument("-d", "--DEBUG", help="Runs the program in debug mode,"
                    "prints verbose output and displays incremental image"
                    "analysis", action="store_true")

parser.add_argument("-i", "--image", help="Path to Hero Roster Screen shot")

parser.add_argument("-r", "--rebuild", help="Rebuild hero database from"
                    "source images", action="store_true")

parser.add_argument("-v", "--verbose", help="Increase verbosity of output"
                    "from image processing", action='count', default=0)
parser.add_argument("-p", "--parallel", help="Utilize as many cores as"
                    "possible while processing",
                    choices=["True", "False"], default="False")
parser.add_argument("-t", "--truth", help="Argument to pass in a truth value"
                    "to file being ran",
                    action="store_true")

args = parser.parse_args()

TRUTH = args.truth
DEBUG = args.DEBUG
REBUILD = args.rebuild
PARALLEL = True if args.parallel.lower() in ["true"] else False
image_ss = None
IMAGE_SS_NAME = None
VERBOSE_LEVEL = args.verbose
try:
    ARCHITECTURE = os.environ["BUILD_TYPE"]
    ARCHITECTURE_TYPES = {"CUDA": "cuda",
                          "CPU": "cpu"}
    ARCHITECTURE = ARCHITECTURE_TYPES[ARCHITECTURE]
except KeyError as e:
    raise EnvironmentError(
        "Environment variable 'BUILD_TYPE' not set. Please set BUILD_TYPE to a"
        " valid option listed in the README") from e

THREADS: dict[str, threading.Thread] = {}

MODEL = None
BORDER_MODEL = None
IMAGE_DB: "imageSearchDB.imageSearch" = None

if VERBOSE_LEVEL == 0:
    logging.disable(logging.INFO)

# Stores cached function results
CACHED = {}

if args.image:
    image_ss = load.load_image(args.image)
    IMAGE_SS_NAME = os.path.basename(args.image)

base_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)))
database_path = os.path.join(base_dir, "database")
database_hero_validation_path = os.path.join(database_path, "temp_image")

# Path to database hero_icons used to build Flann
database_icon_path = os.path.join(database_path, "hero_icon")
flann_path = os.path.join(database_path, "baseHeroes.flann")
database_pickle = os.path.join(database_path, 'imageDB.pickle')

stamina_templates_path = os.path.join(database_path, "stamina_templates")

afk_dir = os.path.join(base_dir, "afk")
models_dir = os.path.join(base_dir, "models")

# Tests directory
tests_dir = os.path.join(base_dir, os.path.pardir, "tests")

# AFK SI Paths
si_path = os.path.join(afk_dir, "si")
si_base_path = os.path.join(si_path, "base")

# AFK Fi Paths
fi_path = os.path.join(afk_dir, "fi")
fi_base_path = os.path.join(fi_path, "base")
fi_train_path = os.path.join(fi_path, "train")

# Yolov5(Stars, FI) and Detectron(Ascension) Model output
final_models_dir = os.path.join(models_dir, "final_models")
yolov5_dir = os.path.join(models_dir, "yolov5")

# Database lvl training data
lvl_path = os.path.join(database_path, "levels")
