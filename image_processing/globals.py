import os
import argparse
import logging
import threading
import pathlib
import shlex

from typing import TYPE_CHECKING

import numpy
import torch

from detectron2.engine import DefaultPredictor

import image_processing.helpers.load_images as load

if TYPE_CHECKING:
    from image_processing.database.image_database import ImageSearch


parser = argparse.ArgumentParser(description='AFK arena object extraction and '
                                 'image analysis.')
parser.add_argument("-d", "--DEBUG", help="Runs the program in debug mode,"
                    "prints verbose output and displays incremental image"
                    "analysis", action="store_true")

parser.add_argument("image_path", metavar="path/to/image",
                    type=str, help="Relative or Absolute Path to Hero Roster Screen shot")

parser.add_argument("-r", "--rebuild", help="Rebuild hero database from"
                    "source images", action="store_true")

parser.add_argument("-v", "--verbose", help="Increase verbosity of output"
                    "from image processing", action='count', default=0)
parser.add_argument("-p", "--parallel", help="Utilize as many cores as"
                    "possible while processing",
                    action="store_true")
parser.add_argument("-t", "--truth", help="Argument to pass in a truth value"
                    "to file being ran",
                    action="store_true")


ARGS: argparse.Namespace = None
TRUTH: bool = None
DEBUG: bool = None
REBUILD: bool = None
PARALLEL: bool = None
IMAGE_SS: numpy.ndarray = None
IMAGE_SS_NAME: str = None
VERBOSE_LEVEL: int = None
MODEL: torch.Tensor = None
BORDER_MODEL: DefaultPredictor = None
IMAGE_DB: "ImageSearch" = None

# Stores cached function results
CACHED = {}
THREADS: dict[str, threading.Thread] = {}


def verbosity(verbose_level: int) -> bool:
    """
    Check if verbose level is verbosity level is higher than the level passed
        in. Provides an easy callable to code that should only be accessible
        when the passed verbose_level is of a certain level.

    Args:
        verbose_level (int): needed level of verbosity

    Returns:
        [bool]: True if verbosity passed in meets the global verbosity levels,
            False otherwise
    """
    return VERBOSE_LEVEL >= verbose_level


def global_parse_args(arg_string: str = None):
    """
    Function to load global arguments from either arg_string or sys.argv.
    The results of the parsing are stored in global argument `ARGS`

    Args:
        arg_string (str, optional): string to parse into command line arguments. Defaults to None.
    """
    global ARGS, TRUTH, DEBUG, REBUILD, PARALLEL, IMAGE_SS, IMAGE_SS_NAME, VERBOSE_LEVEL  # pylint: disable=global-statement
    if arg_string is not None:
        parsed_args = shlex.split(arg_string)
    else:
        parsed_args = None
    ARGS = parser.parse_args(args=parsed_args)

    TRUTH = ARGS.truth
    DEBUG = ARGS.DEBUG
    REBUILD = ARGS.rebuild
    PARALLEL = ARGS.parallel
    VERBOSE_LEVEL = ARGS.verbose

    IMAGE_SS = load.load_image(ARGS.image_path)
    IMAGE_SS_NAME = os.path.basename(ARGS.image_path)
    reload_globals()


def reload_globals():
    """
    Process global variables after they are loaded/reloaded
    """
    # global VERBOSE_LEVEL  # pylint: disable=global-statement
    if VERBOSE_LEVEL == 0:
        logging.disable(logging.INFO)


global_parse_args()

try:
    ARCHITECTURE = os.environ["BUILD_TYPE"]
    ARCHITECTURE_TYPES = {"CUDA": "cuda",
                          "CPU": "cpu"}
    ARCHITECTURE = ARCHITECTURE_TYPES[ARCHITECTURE]
except KeyError as e:
    raise EnvironmentError(
        "Environment variable 'BUILD_TYPE' not set. Please set BUILD_TYPE to a"
        " valid option listed in the README") from e


############################
# All of the following Global variables are dynamically built paths to
# locations within this repo used to load or train objects that are subject to
# changing location during development
############################

# Paths to Image Processing Modules/Directories
GLOBALS_DIR = pathlib.PurePath(os.path.join(os.path.dirname(
    os.path.abspath(__file__))))
DATABASE_DIR = pathlib.PurePath(os.path.join(GLOBALS_DIR, "database"))
AFK_DIR = os.path.join(GLOBALS_DIR, "afk")
MODELS_DIR = os.path.join(GLOBALS_DIR, "models")

# Paths to data inside the Database directory
HERO_ICON_DIR = pathlib.PurePath(os.path.join(DATABASE_DIR, "hero_icon"))
DATABASE_FLAN_PATH = pathlib.PurePath(
    os.path.join(DATABASE_DIR, "IMAGE_DB.flann"))
DATABASE_PICKLE_PATH = pathlib.PurePath(
    os.path.join(DATABASE_DIR, 'IMAGE_DB.pickle'))
DATABASE_LEVELS_DATA_DIR = pathlib.PurePath(
    os.path.join(DATABASE_DIR, "levels"))
DATABASE_STAMINA_TEMPLATES_DIR = pathlib.PurePath(
    os.path.join(DATABASE_DIR, "stamina_templates"))
DATABASE_HERO_VALIDATION_DIR = pathlib.PurePath(
    os.path.join(DATABASE_DIR, "temp_images"))
SEGMENTED_HEROES_DIR = os.path.join(
    DATABASE_HERO_VALIDATION_DIR, "segmented_heroes")

# Tests directory
TESTS_DIR = os.path.join(GLOBALS_DIR, os.path.pardir, "tests")

# AFK/SI Module paths
SI_DIR = os.path.join(AFK_DIR, "si")
SI_TEMPLATE_DIR = os.path.join(SI_DIR, "signature_item_icon")

# AFK/Fi Module paths
FI_DIR = os.path.join(AFK_DIR, "fi")
FI_TEMPLATE_DIR = os.path.join(FI_DIR, "furniture_icons")

# Yolov5(Stars, FI) and Detectron(Ascension) Model output
FINAL_MODELS_DIR = os.path.join(MODELS_DIR, "final_models")
YOLOV5_DIR = os.path.join(MODELS_DIR, "yolov5")
