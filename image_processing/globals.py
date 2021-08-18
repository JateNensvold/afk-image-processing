import os
import cv2
import argparse
import logging

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
image_ss_name = None
VERBOSE_LEVEL = args.verbose

if VERBOSE_LEVEL == 0:
    logging.disable(logging.INFO)


# Stores cached function results
CACHED = {}

if args.image:
    multiplier = 1
    image_ss = cv2.imread(args.image)
    image_ss = cv2.resize(
        image_ss, (image_ss.shape[1]*multiplier, image_ss.shape[0]*multiplier))
    image_ss_name = os.path.basename(args.image)
# DEBUG = True
# DEBUG = False


baseDir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)))
database_path = os.path.join(baseDir, "database")
database_hero_validation_path = os.path.join(database_path, "hero_validation")

database_icon_path = os.path.join(database_path, "hero_icon")
# database_png_path = os.path.join(database_path, "hero_icon/*png"
flannPath = os.path.join(database_path, "baseHeroes.flann")
database_pickle = os.path.join(database_path, 'imageDB.pickle')

staminaTemplatesPath = os.path.join(database_path, "stamina_templates")
# levelTemplatesPath = os.path.join(baseDir, "level_templates")

afk_dir = os.path.join(baseDir, "afk")

siPath = os.path.join(afk_dir, "si")
siTrainPath = os.path.join(siPath, "train")
siBasePath = os.path.join(siPath, "base")

fiPath = os.path.join(afk_dir, "fi")
fi_base_path = os.path.join(fiPath, "base")
fi_train_path = os.path.join(fiPath, "train")
yolov5_dir = os.path.join(fiPath, "fi_detection/yolov5")


lvlPath = os.path.join(database_path, "levels")
