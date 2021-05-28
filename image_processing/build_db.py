import cv2
import os
import pickle

import image_processing.load_images as load
import image_processing.globals as GV
import dill


def pickle_trick(obj, max_depth=10):
    output = {}

    if max_depth <= 0:
        return output

    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError) as e:
        failing_children = []

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)

        output = {
            "fail": obj,
            "err": e,
            "depth": max_depth,
            "failing_children": failing_children
        }

    return output


def buildDB(enrichedDB: bool = False):
    """
    Build and save a new hero database
    Args:
        enrichedDB: flag to add every hero to the database a second time with
            0.15 of their image removed one each side
    Return:
        a load.imageSearch() object filled with heroes
    """
    files = load.findFiles(GV.databaseHeroesPath)
    baseImages = []
    for i in files:
        hero = cv2.imread(i)
        name = os.path.basename(i)
        baseImages.append((name, hero))

    imageDB = load.build_flann(baseImages)

    if enrichedDB:
        croppedImages = load.crop_heroes([i[1] for i in baseImages])
        for index, cropIMG in enumerate(croppedImages):
            imageDB.add_image(baseImages[index][0], cropIMG)

    with open('imageDB.pickle', 'wb') as handle:
        dill.dump(imageDB, handle)
    print(len(imageDB.names))
    return imageDB


def loadDB(Refresh: bool):
    """
    Load/Refresh hero database

    Return:
        load.
    """
    db = None
    with open('imageDB.pickle', 'rb') as handle:
        db = pickle.load(handle)
    print(dir(db))
    return db


if __name__ == "__main__":
    buildDB()
