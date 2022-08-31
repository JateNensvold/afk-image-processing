"""
File that stores the Config options for the afk_image_processing package
When imported will automatically provide a CLI through python argparse for
initializing all the required information for running AFK Arena Roster
Image Detection

Raises:
    EnvironmentError: Raised when the environment variable BUILD_TYPE is not
        set to a valid build option or not set at all. Refer to the Readme.MD
        at the root of this Repo to find valid BUILD_TYPE's
"""
import os
import argparse
import logging
import threading
import pathlib
import shlex

from typing import TYPE_CHECKING, List, Union

import numpy
import torch

import image_processing.utils.load_images as load

from image_processing.utils.timer import Timer
from image_processing.processing.image_processing import HSVRange
from image_processing.database.engravings_database import EngravingSearch

if TYPE_CHECKING:
    from image_processing.database.image_database import ImageSearch

parser = argparse.ArgumentParser(description='AFK arena object extraction and '
                                 'image analysis.')
parser.add_argument("-d", "--DEBUG", help="Runs the program in debug mode,"
                    "prints verbose output and displays incremental image"
                    "analysis", action="store_true")

parser.add_argument(
    "image_path", metavar="path/to/image", type=str,
    help="Relative or Absolute Path to Hero Roster Screen shot")

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
VERBOSE_LEVEL: int = 0

FI_SI_STAR_MODEL: torch.Tensor = None
ASCENSION_BORDER_MODEL: torch.Tensor = None
IMAGE_DB: "ImageSearch" = None
ENGRAVING_DB: EngravingSearch = None

ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
HERO_PORTRAIT_SIZE = 512
MODEL_IMAGE_SIZE = 416
GLOBAL_TIMER = None


# Stores cached function results
CACHED = {}
THREADS: dict[str, threading.Thread] = {}

ZMQ_HOST = "127.0.0.1"
ZMQ_PORT = "5555"


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


def global_parse_args(arg_string: Union[str, List[str]] = None):
    """
    Function to load global arguments from either arg_string or sys.argv.
    The results of the parsing are stored in global argument `ARGS`

    Args:
        arg_string (str, optional): string to parse into command line
            arguments, loads sys.argv when this is None. Defaults to None.
    """
    global ARGS, TRUTH, DEBUG, REBUILD, PARALLEL, IMAGE_SS, IMAGE_SS_NAME, VERBOSE_LEVEL  # pylint: disable=global-statement
    if isinstance(arg_string, str):
        parsed_args = shlex.split(arg_string)
    elif isinstance(arg_string, list):
        parsed_args = arg_string
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
    global GLOBAL_TIMER
    if VERBOSE_LEVEL == 0:
        logging.disable(logging.INFO)

    GLOBAL_TIMER = Timer()


try:
    ARCHITECTURE = os.environ["BUILD_TYPE"]
    ARCHITECTURE_TYPES = {"CUDA": "cuda",
                          "CPU": "cpu"}
    ARCHITECTURE = ARCHITECTURE_TYPES[ARCHITECTURE]
except KeyError as exception_handle:
    raise EnvironmentError(
        "Environment variable 'BUILD_TYPE' not set. Please set BUILD_TYPE to a"
        " valid option listed in the README.MD") from exception_handle


############################
# All of the following Global variables are dynamically built paths to
# locations within this repo used to load or train objects that are subject to
# changing location during development
############################

# Paths to Image Processing Modules/Directories
GLOBALS_DIR = pathlib.Path(os.path.join(os.path.dirname(
    os.path.abspath(__file__))))
DATABASE_DIR = pathlib.Path(os.path.join(GLOBALS_DIR, "database"))
AFK_DIR = os.path.join(GLOBALS_DIR, "afk")
MODELS_DIR = os.path.join(GLOBALS_DIR, "models")

# Paths to data inside the Database directory
# HERO_ICON_DIR = pathlib.Path(os.path.join(DATABASE_DIR, "hero_icon"))
HERO_PORTRAIT_DIRECTORIES: List[pathlib.Path] = []
IMAGE_PROCESSING_PORTRAITS = pathlib.Path(
    os.path.join(DATABASE_DIR, "images", "heroes"))
HERO_PORTRAIT_DIRECTORIES.append(IMAGE_PROCESSING_PORTRAITS)

DATABASE_FLAN_PATH = pathlib.Path(
    os.path.join(DATABASE_DIR, "IMAGE_DB.flann"))
DATABASE_PICKLE_PATH = pathlib.Path(
    os.path.join(DATABASE_DIR, 'IMAGE_DB.pickle'))
DATABASE_LEVELS_DATA_DIR = pathlib.Path(
    os.path.join(DATABASE_DIR, "levels"))
DATABASE_STAMINA_TEMPLATES_DIR = pathlib.Path(
    os.path.join(DATABASE_DIR, "stamina_templates"))
DATABASE_HERO_VALIDATION_DIR = pathlib.Path(
    os.path.join(DATABASE_DIR, "temp_images"))
SEGMENTED_HEROES_DIR = os.path.join(
    DATABASE_HERO_VALIDATION_DIR, "segmented_heroes")
CONFIG_DIR = os.path.join(DATABASE_DIR, "configs")
ENGRAVING_JSON_PATH = os.path.join(CONFIG_DIR, "engraving_values.json")
ASCENSION_JSON_PATH = os.path.join(CONFIG_DIR, "ascension_values.json")

# AFKBuilder Submodule
AFK_Builder_dir = pathlib.Path(os.path.join(DATABASE_DIR, "AFKBuilder"))
AFK_BUILDER_PORTRAITS = pathlib.Path(os.path.join(
    AFK_Builder_dir, "public", "img", "portraits"))
HERO_PORTRAIT_DIRECTORIES.append(AFK_BUILDER_PORTRAITS)

# Tests directory
TESTS_DIR = pathlib.Path(os.path.join(
    GLOBALS_DIR, os.path.pardir, "tests"))

# AFK/SI Module paths
SI_DIR = pathlib.Path(os.path.join(AFK_DIR, "si"))
SI_TEMPLATE_DIR = pathlib.Path(os.path.join(SI_DIR, "signature_item_icon"))

# AFK/Fi Module paths
FI_DIR = pathlib.Path(os.path.join(AFK_DIR, "fi"))
FI_TEMPLATE_DIR = pathlib.Path(os.path.join(FI_DIR, "furniture_icons"))

# Yolov5(Stars, FI) and Yolov5(Ascension) Model output
FINAL_MODELS_DIR = pathlib.Path(os.path.join(MODELS_DIR, "final_models"))
YOLOV5_DIR = pathlib.Path(os.path.join(MODELS_DIR, "yolov5"))

# Path to Yolov5 models
FI_SI_STARS_MODEL_PATH = pathlib.Path(
    os.path.join(FINAL_MODELS_DIR, "fi_si_star_model.pt"))
ASCENSION_BORDER_MODEL_PATH = pathlib.Path(
    os.path.join(FINAL_MODELS_DIR, "hero_ascension_model.pt"))


ENGRAVING_DB = EngravingSearch.from_json(ENGRAVING_JSON_PATH)

HERO_PORTRAIT_OUTLINE_HSV = HSVRange(4, 69, 83, 23, 255, 255)
HERO_ROSTER_HSV = HSVRange(0, 0, 0, 179, 255, 192)
ASCENSION_STAR_HSV = HSVRange(0, 0, 167, 179, 240, 255)
MATRIX_ROW_SPACING_PERCENT = 0.1
