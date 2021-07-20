import pytest
import cv2
import json
import os
import image_processing.build_db as BD
import tests.data
import tests.data.build_data as b_data
import re
import collections


DATA_DIR = tests.data.DIR
JSON_FILE = os.path.join(DATA_DIR, "test_data.json")
TEST_ID_LIST = []


def read_config(config=JSON_FILE):
    """
    All json files to be tested should be passed as params

    """
    with open(config, "r") as f:
        return json.load(f)


def hero_data():
    config = read_config()
    image_db = BD.get_db()
    json_test_data = config
    for image_path, si_info in json_test_data.items():
        image = cv2.imread(image_path)
        json_dict = {}
        image_name = os.path.basename(
            image_path)
        b_data.generate_data(json_dict, image_name, image, image_db)
        gen_heroes = json_dict[image_name]["heroes"]
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
                    gen_hero_info = collections.defaultdict(lambda: None)
                    test_hero_info = collections.defaultdict(lambda: None)
                    if isinstance(gen_heroes[
                            _row_index][_hero_index], list):
                        gen_hero_data = gen_heroes[_row_index][_hero_index][0]
                    else:
                        gen_hero_data = gen_heroes[_row_index][_hero_index]

                    if isinstance(test_heroes[
                            _row_index][_hero_index], list):
                        test_hero_data = test_heroes[
                            _row_index][_hero_index][0]
                    else:
                        test_hero_data = test_heroes[_row_index][_hero_index]

                    gen_hero_regex = re.search(
                        r"([a-zA-Z]*) (\d{2})(\d{1})", gen_hero_data)
                    test_hero_regex = re.search(
                        r"([a-zA-Z]*) (\d{2})(\d{1})", test_hero_data)
                    if gen_hero_regex is not None:
                        gen_hero_info["name"] = gen_hero_regex.group(1)
                        gen_hero_info["si"] = gen_hero_regex.group(2)
                        gen_hero_info["fi"] = gen_hero_regex.group(3)
                    if isinstance(gen_heroes[
                            _row_index][_hero_index], list):
                        gen_hero_info["raw"] = gen_heroes[
                            _row_index][_hero_index][1]

                    if test_hero_regex is not None:
                        test_hero_info["name"] = test_hero_regex.group(1)
                        test_hero_info["si"] = test_hero_regex.group(2)
                        test_hero_info["fi"] = test_hero_regex.group(3)
                    else:
                        test_hero_info["name"] = "n/a"
                        test_hero_info["si"] = "n/a"
                        test_hero_info["fi"] = "n/a"
                    global TEST_ID_LIST
                    TEST_ID_LIST.append("{}({})({})".format(
                        image_name, _row_index, _hero_index))

                    yield gen_hero_info, test_hero_info
        except IndexError as e:
            # print("Gen heroes: {}".format(gen_heroes))
            for _i in gen_heroes:
                print([_hero_object[0] for _hero_object in _i])
            raise IndexError("{} gen_len: {} test_len: {} index: {}".format(
                image_name,
                len(gen_heroes),
                len(test_heroes),
                _row_index)) from e


def get_id():
    for _id in TEST_ID_LIST:
        yield _id


def image_data():
    config = read_config()
    json_test_data = config
    for image_path, si_info in json_test_data.items():
        image_name = os.path.basename(
            image_path)
        yield image_name


@ pytest.mark.parametrize("hero_info", hero_data(), ids=get_id())
def test_si(hero_info):
    gen_si = hero_info[0]["si"]
    base_si = hero_info[1]["si"]
    assert base_si == gen_si, (base_si, gen_si, dict(hero_info[0]))
