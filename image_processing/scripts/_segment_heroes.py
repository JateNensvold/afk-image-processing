import os
import time
import cv2

import image_processing.afk.si.get_si as GS
import image_processing.build_db as BD
import image_processing.globals as GV


def parse_dict(json_dict: dict, hero_dict: dict, hero_dir: str, file_name: str):
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
    if not os.path.exists(hero_dir):
        os.mkdir(hero_dir)

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

            si_fi = f"{hero_si}{hero_fi}"
            hero_si_fi_directory = os.path.join(
                hero_name_directory, si_fi)
            if si_fi not in os.listdir(hero_name_directory):
                os.mkdir(hero_si_fi_directory)

            image_save_name = f"{hero_name}_{hero_ascension}_{hero_si}{hero_fi}"
            hero_fi_si_directory_files = os.listdir(hero_si_fi_directory)
            count = 0
            for file_names in hero_fi_si_directory_files:
                if image_save_name in file_names:
                    count += 1
            image_save_name = f"{image_save_name}_{count}"
            image_save_path = os.path.join(
                hero_si_fi_directory, image_save_name)
            image_save_path = f"{image_save_path}.jpg"
            hero_name = f"{''.join(hero_object)}_{hero_count}"
            print(image_save_path)

            while hero_name not in hero_dict:
                hero_count += 1
                hero_name = f"{hero_object}_{hero_count}"

            cv2.imwrite(image_save_path,
                        hero_dict[hero_name]["image"])
            hero_count += 1


if __name__ == "__main__":
    imageDB = BD.get_db(enriched_db=True)

    if GV.TRUTH:


        for image_file_name in os.listdir(GV.ATABASE_HERO_VALIDATION_PATH):
            print(f"Starting {image_file_name}")
            start_time = time.time()
            if image_file_name.endswith(".png") or image_file_name.endswith(".jpg"):
                file_path = os.path.join(GV.HERO_VALIDATION_DIR, image_file_name)
                image = cv2.imread(file_path)
                image_hero_dict = {}
                hero_json_dict = GS.get_si(
                    image, image_file_name, imageDB=imageDB, hero_dict=image_hero_dict,
                    faction=True)
                image_hero_dict = image_hero_dict["hero_dict"]
                parse_dict(hero_json_dict, image_hero_dict, GV.SEGMENTED_HEROES_DIR,
                           image_file_name)
            end_time = time.time()

            print(f"Finished {image_file_name} in {end_time-start_time}")
    else:
        # When running in single image mode(i.e. GV.TRUTH is false) pass image
        #   in with GV.image_ss(i.e. --image/-i)
        image = GV.image_ss
        image_hero_dict = {}
        hero_json_dict = GS.get_si(
            image, GV.IMAGE_SS_NAME, imageDB=imageDB, hero_dict=image_hero_dict,
            faction=True)
        image_hero_dict = image_hero_dict["hero_dict"]
        parse_dict(hero_json_dict, image_hero_dict, "./temp", GV.IMAGE_SS_NAME)


# python3 _segment_heroes.py -t
