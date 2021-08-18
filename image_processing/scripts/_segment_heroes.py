import os
import cv2
import time

import image_processing.afk.si.get_si as GS
import image_processing.build_db as BD
import image_processing.globals as GV

HERO_VALIDATION_DIR = GV.database_hero_validation_path


def parse_dict(json_dict, hero_dict, hero_dir, file_name):
    """
    Accept a dictionary with hero results and generate directories under
        database/hero_validation/sorted_heroes for all segmented images/heros
        detected from source image
    Args:
        json_dict: dictionary returned by GS.get_si
        hero_dict: dictionary/arg 0 returned by
            image_processing.processing.getHeroes
        hero_dir: directory under database/hero_validation to generate new
            directories in
        file_name: name of source image that json_dict was generated from,
            used to access data from json_dict that was generated from it.
    Return:
        None
    """
    hero_count = 0
    for row in json_dict[file_name]["heroes"]:
        for hero_object in row:
            hero_name = hero_object[0]
            hero_si = hero_object[1]
            hero_fi = hero_object[2]
            hero_ascension = hero_object[3]
            faction = hero_object[4]

            faction_directory = os.path.join(
                hero_dir, faction)
            if faction not in os.listdir(
                    hero_dir):
                os.mkdir(faction_directory)

            hero_name_directory = os.path.join(
                faction_directory, hero_name)
            if hero_name not in os.listdir(
                    faction_directory):
                os.mkdir(hero_name_directory)

            si_fi = "{}{}".format(hero_si, hero_fi)
            hero_si_fi_directory = os.path.join(
                hero_name_directory, si_fi)
            if si_fi not in os.listdir(hero_name_directory):
                os.mkdir(hero_si_fi_directory)

            image_save_name = "{}_{}_{}{}".format(
                hero_name, hero_ascension, hero_si, hero_fi)
            hero_fi_si_directory_files = os.listdir(hero_si_fi_directory)
            count = 0
            for file_names in hero_fi_si_directory_files:
                if image_save_name in file_names:
                    count += 1
            image_save_name = "{}_{}".format(
                image_save_name, count)
            image_save_path = os.path.join(
                hero_si_fi_directory, image_save_name)
            image_save_path = "{}.jpg".format(image_save_path)
            hero_name = "{}_{}".format(
                "".join(hero_object), hero_count)
            print(image_save_path)
            cv2.imwrite(image_save_path,
                        hero_dict[hero_name]["image"])
            hero_count += 1


if __name__ == "__main__":
    imageDB = BD.get_db(enrichedDB=True)

    build_database = True

    if GV.TRUTH:
        sorted_heroes_directory = os.path.join(
            HERO_VALIDATION_DIR, "sorted_heroes")

        for file_name in os.listdir(HERO_VALIDATION_DIR):
            print("Starting {}".format(file_name))
            start_time = time.time()
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                file_path = os.path.join(HERO_VALIDATION_DIR, file_name)
                image = cv2.imread(file_path)
                hero_dict = {}
                json_dict = GS.get_si(
                    image, file_name, imageDB=imageDB, hero_dict=hero_dict,
                    faction=True)
                hero_dict = hero_dict["hero_dict"]
                parse_dict(json_dict, hero_dict, sorted_heroes_directory,
                           file_name)
            end_time = time.time()

            print("Finished {} in {}".format(file_name, end_time-start_time))
    else:
        image_path = "/home/nate/downloads/IMG_20210814_220011.jpg"
        image = cv2.imread(image_path)
        hero_dict = {}
        json_dict = GS.get_si(
            image, image_path, imageDB=imageDB, hero_dict=hero_dict,
            faction=True)
        hero_dict = hero_dict["hero_dict"]
        parse_dict(json_dict, hero_dict, "./temp", image_path)
