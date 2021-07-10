import pytest
import cv2
import json
import os
import image_processing.build_db as BD
import tests.data
import tests.data.build_data as b_data

# TEST_TYPE_DIR = os .path.dirname(os.path.abspath(__file__))
# SI_DIR = os.path.abspath(os.path.join(TEST_TYPE_DIR, os.pardir))
# DATA_DIR = os.path.abspath(os.path.join(
#     os.path.join(TEST_TYPE_DIR, os.pardir), "data"))

DATA_DIR = tests.data.DIR
JSON_FILE = os.path.join(DATA_DIR, "test_data.json")

# @pytest.fixture(params=[JSON_FILE])


def read_config(config=JSON_FILE):
    """
    All json files to be tested should be passed as params

    """
    with open(config, "r") as f:
        return json.load(f)


# @pytest.fixture(params=read_config)
def si_data():
    # print(read_config)
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
        for _row_index in range(max(len(gen_heroes), len(test_heroes))):
            for _hero_index in range(max(len(gen_heroes[_row_index]),
                                         len(test_heroes[_row_index]))):
                yield gen_heroes[_row_index][_hero_index], \
                    test_heroes[_row_index][_hero_index],
                # for _row in si_info["heroes"]:
                #     for _hero in _row:
                #         yield _hero

                # @pytest.fixture()
                # def get_data(image_data):
                #     return image_data


def pytest_generate_test(metafunc):
    if "image_data" in metafunc.fixturenames:
        si_data = metafunc.config.hero_si_data
        metafunc.parametize("si_data", si_data)


@pytest.mark.parametrize("si_info", si_data())
def test_si(si_info):
    print("si_info", si_info)
    assert 1 == 0
