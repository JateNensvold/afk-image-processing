import re
import collections

import cv2
import numpy as np

import image_processing.globals as GV
import image_processing.load_images as load
import image_processing.afk.hero_object as HO


FLANN_INDEX_KDTREE = 1


class ImageSearch():
    """
    Wrapper around an in memory image database

    Uses flann index to store and search for images base on extracted
        SIFT features

    Args:
        lowes_ratio: ratio to apply to all feature matches(0.0 - 1.0), higher
            is less unique
                (https://stackoverflow.com/questions/51197091/
                how-does-the-lowes-ratio-test-work)
    """

    def __init__(self, lowes_ratio: int = 0.8):
        self.ratio = lowes_ratio
        index_param = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        search_param = {"checks": 50}
        patch_size = 16

        self.matcher = cv2.FlannBasedMatcher(index_param, search_param)
        # self.matcher = cv2.BFMatcher(cv2.NORM_L1)
        self.extractor = cv2.SIFT_create(edgeThreshold=patch_size)

        # self.extractor = cv2.ORB_create(
        #     edgeThreshold=patchSize, patchSize=patchSize)
        self.image_data = {}
        self.names = {}

    def __getstate__(self):

        file_name = GV.DATABASE_FLAN_PATH
        self.matcher.write(file_name)
        return {"images": self.image_data, "names": self.names,
                "matcher": file_name,
                "descriptors": self.matcher.getTrainDescriptors()
                }

    def __setstate__(self, incoming):
        self.matcher: cv2.FlannBasedMatcher
        self.__init__()
        self.image_data = incoming["images"]
        self.names = incoming["names"]
        descriptors = incoming["descriptors"]
        for des in descriptors:
            self.matcher.add([des])

        self.matcher.read(incoming["matcher"])

    def get_good_features(self, matches: list, ratio: int):
        """
        Return a list of "good" features from 'matches' that pass the Lowe's
            ratio test at the ratio passed in
        https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html

        Args:
            matches: list of keypoint descriptors
            ratio: integer representing what ratio Lowes' test must pass,
                default ratio is 0.8

        Return:
            list of 'matches' that pass the Lowe's ratio test at the given
                ratio
        """
        good_features = []
        for best_match, second_best_match in matches:
            if best_match.distance < ratio * second_best_match.distance:
                good_features.append(best_match)

        return good_features

    def add_image(self, hero_info: HO.hero_object, hero: np.array):
        """
        Adds image features to image database and stores image
            in internal list for later use

        Args:
            name: image_processing.hero_object.hero_object representing hero
                info
            hero: image to add to database
        Return:
            None
        """
        width, height = hero.shape[:2]
        hero = cv2.resize(hero, (height*3, width*3))
        _kp, des = self.extractor.detectAndCompute(hero, None)

        self.matcher.add([des])

        name_regex_results = re.split(r"(-|_|\.)", hero_info.name.lower())
        clean_name = name_regex_results[0]
        counter = len(self.names)

        self.names[counter] = clean_name
        if clean_name not in self.image_data:
            self.image_data[clean_name] = {}
        self.image_data[clean_name][counter] = hero
        self.image_data[clean_name]["info"] = hero_info
        if GV.VERBOSE_LEVEL >= 2:
            print(f"Hero: {hero_info.name} Faction: {hero_info.faction}")

    def search(self, hero: np.array,
               min_features: int = 5,
               crop_hero: bool = True):
        """
        Find closest matching image in the database

        Args:
            hero: cv2 image to find a match for
            display: flag to show base image and search image for humans to
                compare/confirm that the search was successful
            min_features: minimum number of both "good_features" and single
                hero votes to attemp to find on a search
            crop_hero: flag to crop the default amount from the border of the
                hero image passed in
        Returns:
            Tuple containing
                (hero_info: image_processing.hero_object.hero_object,
                 image: np.ndarray)
            of the closest matching image in the database
        """
        if crop_hero:
            cropped_heroes_list = load.crop_heroes(
                [hero], 0.15, 0.08, 0.25, 0.2)
            hero = cropped_heroes_list[0]

        _kp, des = self.extractor.detectAndCompute(hero, None)
        matches = self.matcher.knnMatch(np.array(des), k=2)

        ratio = self.ratio
        good_features = []
        hero_result = []

        while len(good_features) < min_features and ratio <= 1.0:
            good_features = self.get_good_features(matches, ratio+0.05)
            ratio += 0.05

        assert (len(good_features) >= min_features), \
            (f"Failed to find enough \"good\" features ({len(good_features)}) "
             "from database to match a similar image. Expected at least "
             f"({min_features}) good features to be found")
        # if len(good_features) < min_features:
        #     good_features = matches[0]

        hero_result = self._search_results(good_features)
        best_match_raw = hero_result[0]
        best_match_dict = best_match_raw[1]
        best_match_name = best_match_raw[0]

        best_match_image: np.ndarray = self.image_data[best_match_name][
            max(best_match_dict["base"], key=best_match_dict["base"].get)]
        best_match_info: HO.hero_object = self.image_data[best_match_name][
            "info"]

        return (best_match_info, best_match_image)

    def _search_results(self, good_features: list):
        """
        Organises good_feature matches by hero that each keypoint descriptor
            matched to
        Args:
            good_features: keypoint descriptors that have passed the lowe's
                ratio test.

        Return:
            List of tuples of dictionaries where list is sorted by
                number(decending) of total keypoint matches to a hero and
                dictionary is the following
            dictionary of (name: {"total": total keypoint matches,
                                 "base": keypoint matches to single image,
                                 "distance": distance between keypoint
                                            descriptor and database point})
        """

        matches_idx = [x.imgIdx for x in good_features]
        index_counter = {}
        for _match_index, _idx in enumerate(matches_idx):
            if _idx not in index_counter:
                index_counter[_idx] = []
            index_counter[_idx].append(_match_index)
        counter = collections.Counter(matches_idx)
        heroes = {}
        for _counter_index, i in enumerate(counter):
            occurrences = counter[i]
            name = self.names[i]
            if name not in heroes:
                heroes[name] = {}
                heroes[name]["total"] = occurrences
                heroes[name]["base"] = {}
                heroes[name]["base"][i] = occurrences
                heroes[name]["distance"] = [
                    good_features[_i].distance for _i in index_counter[i]]
            else:
                heroes[name]["total"] = heroes[name]["total"] + occurrences
                heroes[name]["distance"] += [good_features[_i].distance for _i
                                             in index_counter[i]]

                if i not in heroes[name]["base"]:
                    heroes[name]["base"][i] = occurrences
                else:
                    heroes[name]["base"][i] = heroes[name]["base"][i] + \
                        occurrences

        # Sort heroes by total number of matches
        output = sorted(
            heroes.items(), key=lambda hero: hero[1]["total"], reverse=True)
        return output


def build_flann(image_list: list[HO.hero_object],
                ratio: int = 0.8) -> "ImageSearch":
    """
    Build database of heroes to match to

    Args:
        image_list: list of (image(np.array), name(str), faction(str)) tuples
            to build database from

    Return:
        An instance of ImageSearch() with image_list added to it with the
            matcher trained on them
    """
    image_database = ImageSearch(lowes_ratio=ratio)
    for _image_info_index, image_info in enumerate(image_list):

        image_database.add_image(image_info, image_info.image)
    image_database.matcher.train()

    return image_database
