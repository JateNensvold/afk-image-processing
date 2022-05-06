import os
import re
import json

import pytest

import tests.data
import tests.data.build_data as b_data
import image_processing.globals as GV
import image_processing.utils.load_models as LM

DATA_DIR = tests.data.DIR
JSON_FILE = os.path.join(DATA_DIR, "test_data.json")
TEST_ID_LIST = []


def read_config(config=JSON_FILE):
    """
    All json files to be tested should be passed as params

    """
    with open(config, "r") as file_handle:
        return json.load(file_handle)


FULL_IMAGE_PATH = None


def hero_data():
    config = read_config()
    json_test_data = config

    LM.load_files(str(GV.FI_SI_STARS_MODEL_PATH))
    count = 0
    for image_path, si_info in json_test_data.items():
        if count >= 3:
            break
        count += 1
        global FULL_IMAGE_PATH
        FULL_IMAGE_PATH = os.path.join(GV.ROOT_DIR, os.path.pardir, image_path)
        GV.global_parse_args(FULL_IMAGE_PATH)
        json_dict = b_data.generate_data(GV.IMAGE_SS_NAME, GV.IMAGE_SS)
        gen_heroes = json_dict["heroes"]
        test_heroes = si_info["heroes"]

        if len(gen_heroes) != len(test_heroes):
            gen_length = len(gen_heroes)
            test_length = len(test_heroes)
            gen_heroes += [
                ["Null"] for _i in range(test_length-gen_length)]
        try:
            for _row_index in range(max(len(gen_heroes), len(test_heroes))):
                if len(gen_heroes[_row_index]) != len(test_heroes[_row_index]):
                    gen_length = len(gen_heroes[_row_index])
                    test_length = len(test_heroes[_row_index])
                    gen_heroes[_row_index] += [
                        "Null" for _i in range(test_length-gen_length)]

                for _hero_index in range(len(test_heroes[_row_index])):
                    gen_hero_data = gen_heroes[_row_index][_hero_index]

                    if isinstance(test_heroes[
                            _row_index][_hero_index], list):
                        test_hero_data = test_heroes[
                            _row_index][_hero_index][0]
                    else:
                        test_hero_data = test_heroes[_row_index][_hero_index]

                    if "food" in test_hero_data.lower():
                        continue
                    yield gen_hero_data, test_hero_data
        except IndexError as exception_handle:
            for _i in gen_heroes:
                print([_hero_object[0] for _hero_object in _i])
            raise IndexError(
                f"{GV.IMAGE_SS_NAME} gen_len: {len(gen_heroes)} test_len: "
                f"{len(test_heroes)} index: {_row_index}") from exception_handle


def get_id():
    for _id in TEST_ID_LIST:
        yield _id


@ pytest.mark.parametrize("hero_info", hero_data(), ids=get_id())
def test_si(hero_info):

    detected_hero_data = hero_info[0]
    test_hero_string = hero_info[1]
    test_hero_regex = re.search(
        r"([a-zA-Z]*) (\d{2})(\d{1})", test_hero_string)

    fail_string = (f"Detected: '{detected_hero_data[0]} "
                   f"{''.join(detected_hero_data[1:3])}' Actually: "
                   f"'{test_hero_string}'")
    detected_raw = detected_hero_data[-1]

    assert test_hero_regex.group(1).lower(
    ) == detected_hero_data[0].lower(), f"Wrong Hero Detected - {fail_string} {FULL_IMAGE_PATH}"
    assert int(test_hero_regex.group(2)) == int(
        detected_hero_data[1]), f"Wrong SI Detected - {fail_string} Raw: {detected_raw['si']} {FULL_IMAGE_PATH}"
    assert int(test_hero_regex.group(3)) == int(
        detected_hero_data[2]), f"Wrong FI Failure - {fail_string} Raw: {detected_raw['fi']} {FULL_IMAGE_PATH}"
