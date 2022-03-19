from typing import Any, Dict, List, TypedDict, Union

import cv2
import numpy as np

from image_processing.processing import SegmentResult
import image_processing.globals as GV
from image_processing.afk.hero.hero_data import HeroImage
from image_processing.load_images import CropImageInfo, crop_heroes, display_image


FLANN_INDEX_KDTREE = 1

image_search_dict = TypedDict("image_search_dict", {
                              "images": Dict[str,
                                             Dict[Union[str, int],
                                                  Union[np.ndarray,
                                                  HeroImage]]],
                              "names": Dict[int, str],
                              "descriptors": Any,
                              "matcher": cv2.FlannBasedMatcher})


class NoMatchException(Exception):
    """_summary_

    Args:
        Exception (_type_): _description_

    Returns:
        _type_: _description_
    """


class ImageDatabaseHero:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, hero_name: str, hero_info: HeroImage,
                 hero_index: int):
        """_summary_

        Args:
            hero_name (str): _description_
            hero_info (HeroImage): _description_
            hero_index (int): _description_
        """
        self.name = hero_name
        self.hero_instances: List[HeroImage] = [hero_info]
        self.hero_index_lookup: Dict[int, HeroImage] = {
            hero_index: hero_info}

    def first(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if len(self.hero_instances) == 0:
            return None
        else:
            return self.hero_instances[0]

    def __str__(self):
        return f"ImageDatabaseHero<{self.name}, image_count={len(self.hero_instances)}>"

    def __repr__(self) -> str:
        return str(self)


class FeatureMatch:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, hero_name: str, hero_index: int, flann_match: cv2.DMatch):
        """_summary_

        Args:
            hero_name (str): _description_
            hero_index (int): _description_
            flann_match (Any): _description_
        """
        self.hero_name = hero_name
        self.hero_index = hero_index
        self.feature = flann_match

    def __str__(self) -> str:
        return f"FeatureMatch<{self.hero_name}, hero_index={self.hero_index}>"


class HeroMatchList:
    """
    Wrapper around a list of "HeroMatch" objects that has several utility
        methods for keeping track of the number of flann matches for each hero.
        By default at all times this data structure should provide a sorted
        list of HeroMatch objects based on decending HeroMatch.match_count
    """

    def __init__(self, matches: List["HeroMatch"]):
        """
        Wrapper around a list of "HeroMatch" objects that has several utility
        methods for keeping track of the number of flann matches for each hero.
        By default at all times this data structure should provide a sorted
        list of HeroMatch objects based on decending HeroMatch.match_count

        Args:
            matches (List[HeroMatch]): list of hero matches to add
        """
        self.hero_matches: List[HeroMatch] = []
        self.hero_dict: Dict[str, HeroMatch] = {}
        self.current_iter = 0

        self.extend(matches)

    def __iter__(self):
        """
        Return object to iterate over

        Returns:
            self
        """
        return self

    def __next__(self):
        """
        Provide the next element to iterate over
        """
        if self.current_iter < len(self.hero_matches):
            self.current_iter += 1
            return self.hero_matches[self.current_iter - 1]
        self._cur_iter = 0
        raise StopIteration

    def __getitem__(self, index: int):
        """
        Fetch the element at index

        Args:
            index (int): Index of list to get
        """
        return self.hero_matches[index]

    def __str__(self):

        return f"HeroMatchList<len={len(self.hero_matches)}, top={self.hero_matches[:3]}>"

    def append(self, match: "HeroMatch", sort: bool = True):
        """
        Adds a single 'HeroMatch' into 'HeroMatch' list
            and then optionally sorts self to maintain the match_count order

        Args:
            match (HeroMatch): match getting added to list
            sort (bool): flag to sort the match list after adding new element
        """
        if match.name in self.hero_dict:
            self.hero_dict[match.name].combine(match)
        else:
            self.hero_dict[match.name] = match
            self.hero_matches.append(match)
        if sort:
            self.sort()

    def sort(self):
        """
        Sort list so matches are ordered by decreasing hero_match.match_count
        """
        self.hero_matches.sort(key=lambda hero_match: hero_match.match_count,
                               reverse=True)

    def extend(self, matches: Union[List["HeroMatch"], "HeroMatchList"]):
        """
        Merge a List[HeroMatch] or 'HeroMatchList' object into self and
            combine all matches that are for the same hero

        Args:
            matches (List[HeroMatch]): matches to get merged in
        """
        for hero_match in matches:
            self.append(hero_match, sort=False)
        self.sort()

    def best(self):
        """
        Returns the HeroMatch with the highest number of matches
        """
        if len(self.hero_matches) > 0:
            return self.hero_matches[0]
        raise NoMatchException("No HeroMatch to return object length is 0")


class HeroMatch:
    """
        A Data structure for keeping track of the number of 'FeatureMatch'
            that a hero has recieved
    """

    def __init__(self, hero_name: str):
        """


        Args:
            hero_name (str): hero name to keep track of FeatureMatch'
        """
        self.name = hero_name
        self.matches: List[FeatureMatch] = []

    def __str__(self):
        """
        String representation of object, that shows hero name and match_count
        """
        return f"HeroMatch<{self.name}, match_count={self.match_count}>"

    def __repr__(self) -> str:
        return str(self)

    @property
    def match_count(self):
        """
        Property method  that calculates number of FeatureMatches for hero
        """
        return len(self.matches)

    def add_match(self, feature_match: FeatureMatch):
        """
        Adds a new flann_match to object
        Args:
            feature_match (FeatureMatch): match getting added
        """
        self.matches.append(feature_match)

    def combine(self, new_match: "HeroMatch"):
        """
        Combine 'new_match' into HeroMatch object

        Args:
            new_match (HeroImage): object getting combined with self
        """

        for match in new_match.matches:
            self.matches.append(match)

    # @classmethod
    # def merge(cls, first_matches: List["HeroMatch"],
    #           second_matches: List["HeroMatch"]):
    #     """_summary_

    #     Args:
    #         first_matches (List[HeroImage]): _description_
    #         second_matches (List[HeroImage]): _description_

    #     Raises:
    #         NoMatchException: _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     match_dict: Dict[str, HeroMatch] = {}

    #     for hero_match in first_matches:
    #         if hero_match.name not in match_dict:
    #             match_dict[hero_match.name] = hero_match
    #         else:
    #             match_dict[hero_match.name].combine(hero_match)

    #     for hero_match in second_matches:
    #         if hero_match.name not in match_dict:
    #             match_dict[hero_match.name] = hero_match
    #         else:
    #             match_dict[hero_match.name].combine(hero_match)

    #     sorted_heroes = sorted(
    #         match_dict.values(),
    #         key=lambda hero_match: hero_match.match_count,
    #         reverse=True)
    #     return sorted_heroes


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
        self.count = 0

        self.matcher = cv2.FlannBasedMatcher(index_param, search_param)
        # self.matcher = cv2.BFMatcher(cv2.NORM_L1)
        self.extractor = cv2.SIFT_create(edgeThreshold=patch_size)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # self.extractor = cv2.ORB_create(
        #     edgeThreshold=patchSize, patchSize=patchSize)
        self.hero_lookup: Dict[str, ImageDatabaseHero] = {}
        self.index_lookup: Dict[int, ImageDatabaseHero] = {}

    def __getstate__(self) -> image_search_dict:
        """
        Serialize the ImageSearch Object into a savable state

        Returns:
            [dict]: Dictionary with "images", "names", "matcher" and
                "descriptors" as keys
        """

        file_name = str(GV.DATABASE_FLAN_PATH)
        self.matcher.write(file_name)
        return {"hero_lookup": self.hero_lookup,
                "index_lookup": self.index_lookup,
                "matcher": file_name,
                "descriptors": self.matcher.getTrainDescriptors()
                }

    def __setstate__(self, incoming: image_search_dict):
        """
        Deserialize the ImageSearch Object into a savable state

        Args:
            incoming ([image_search_dict]): [description]
        """
        self.matcher: cv2.FlannBasedMatcher
        self.__init__()
        self.hero_lookup = incoming["hero_lookup"]
        self.index_lookup = incoming["index_lookup"]
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

    def image_pre_process(self, hero_image: np.ndarray,
                          crop_info: CropImageInfo = None,
                          image_multiplier=1.0) -> np.ndarray:
        """_summary_

        Args:
            hero_image (np.ndarray): _description_
            crop_info (CropImageInfo, optional): _description_. Defaults to None.
            image_multiplier (float, optional): _description_. Defaults to 1.0.

        Returns:
            np.ndarray: _description_
        """

        gray_image = cv2.cvtColor(hero_image, cv2.COLOR_BGR2GRAY)

        width, height = gray_image.shape[:2]
        ascpect_ratio = width/height
        new_width = int(GV.HERO_PORTRAIT_SIZE * image_multiplier)
        image_scale = new_width/width
        new_height = int(height * image_scale)
        resized_image = cv2.resize(
            gray_image, (new_height, new_width))


        if crop_info:
            cropped_heroes_list = crop_heroes([resized_image], crop_info)
            resized_image = cropped_heroes_list[0]


        clahe_image = self.clahe.apply(resized_image)
        display_image([resized_image, clahe_image])

        return clahe_image

    def add_image(self, hero_info: HeroImage,
                  crop_info: CropImageInfo = None):
        """
        Adds image features to image database and stores image
            in internal list for later use

        Args:
            hero_info: (HeroImage): A wrapper around an image that includes the
                image itself, the image name and the location the image was
                loaded from
            crop_info (CropImageInfo): Named Tuple that contains information on
                how much to crop 'hero'. When this is None no cropping happens
        Return:
            None
        """

        hero_image = self.image_pre_process(hero_info.image, crop_info)

        _keypoint, descriptor = self.extractor.detectAndCompute(
            hero_image, None)

        self.matcher.add([descriptor])

        hero_index = len(self.index_lookup)

        if GV.verbosity(2):
            print(f"Added Hero ({hero_index}): {hero_info.name} from "
                  f"{hero_info.image_path} Size: ({hero_image.shape[0]}, "
                  f"{hero_image.shape[1]}) -> {hero_image.shape[:2]} "
                  f"{'(cropped)' if crop_info else ''}")

        if hero_info.name not in self.hero_lookup:
            database_hero = ImageDatabaseHero(
                hero_info.name, hero_info, hero_index)
            self.hero_lookup[hero_info.name] = database_hero
        else:
            database_hero = self.hero_lookup[hero_info.name]
            database_hero.hero_index_lookup[hero_index] = hero_info
            database_hero.hero_instances.append(hero_info)

        self.index_lookup[hero_index] = database_hero

    def search(self, segment_info: SegmentResult,
               min_features: int = 5,
               crop_info: CropImageInfo = CropImageInfo(0.15, 0.08, 0.25, 0.2),
               image_multiplier=1.0):
        """
        Find closest matching image in the database

        Args:
            segment_info: info describing the location of a segmented image
                that represents a subsection of a larger image
            display: flag to show base image and search image for humans to
                compare/confirm that the search was successful
            min_features: minimum number of both "good_features" and single
                hero votes to attemp to find on a search
            crop_info (CropImageInfo): Named Tuple that contains information on
                how much to crop 'hero'. When this is None no cropping happens
        Returns:
            Tuple containing
                (hero_info: image_processing.hero_data.HeroImage,
                 image: np.ndarray)
            of the closest matching image in the database
        """

        hero_image = self.image_pre_process(segment_info.image, crop_info, image_multiplier)

        _keypoint, descriptor = self.extractor.detectAndCompute(
            hero_image, None)
        matches = self.matcher.knnMatch(np.array(descriptor), k=2)

        ratio = self.ratio
        good_features: List[cv2.DMatch] = []

        while len(good_features) < min_features and ratio <= 1.0:
            good_features = self.get_good_features(matches, ratio+0.05)
            ratio += 0.05

        assert (len(good_features) >= min_features), (
            f"Failed to find enough \"good\" features ({len(good_features)}) "
            "from database to match a similar image. Expected at least "
            f"({min_features}) good features to be found")

        hero_matches = self._search_results(good_features)
        if crop_info:
            self.count += 1
            self.count %= 20
        # if self.count in (12, 19):
        #     print(f"{self.count} {hero_matches}\n")
        #     for hero_match in hero_matches:
        #         print(hero_match.name, self.hero_lookup[hero_match.name])
        #         for feature_match in hero_match.matches:
        #             print(
        #                 self.index_lookup[feature_match.hero_index].hero_index_lookup[feature_match.hero_index].image.shape)

        # Check for a better hero match with different image preprocessing or
        #   log diagnostic information about the hero_matches if no better
        #   preprocessing is possible
        if hero_matches.best().match_count < 10 or not crop_info:
            if crop_info:
                if GV.verbosity(1):
                    print(f"\tOriginal {hero_matches}")
                new_matches = self.search(segment_info, crop_info=None)
                hero_matches.extend(new_matches)
            else:
                if GV.verbosity(1):
                    print(f"\tRedo {hero_matches}")

        if self.hero_lookup[hero_matches.best().name].first() is None:
            raise NoMatchException(
                f"Unable to find a match for {hero_matches.best().name}")

        if GV.verbosity(1):
            print(hero_matches.best(), hero_matches)

        return hero_matches

    def _search_results(self, good_features: List[cv2.DMatch]):
        """
        Organises good_feature matches by hero that each keypoint descriptor
            matched to
        Args:
            good_features: keypoint descriptors that have passed the lowe's
                ratio test.

        Return:
            HeroMatchList object containing all HeroMatches extracted
                from good_features
        """

        hero_matches_dict: Dict[str, HeroMatch] = {}
        for flann_match in good_features:
            hero_index: int = flann_match.imgIdx
            hero_name = self.index_lookup[hero_index].name
            feature = FeatureMatch(hero_name, hero_index, flann_match)

            if hero_name not in hero_matches_dict:
                hero_matches_dict[hero_name] = HeroMatch(hero_name)
            hero_matches_dict[hero_name].add_match(feature)

        hero_matches = HeroMatchList(hero_matches_dict.values())

        return hero_matches


def build_flann(image_list: list[HeroImage],
                ratio: int = 0.8,
                enriched_db=False) -> "ImageSearch":
    """
    Build database of heroes to match to

    Args:
        image_list: list of (image(np.array), name(str), faction(str)) tuples
            to build database from
        ratio (int): lowes_ratio: ratio to apply to all
            feature matches(0.0 - 1.0), higher is less unique
            (https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work)
        enriched_db (bool): flag to add every hero to the database a second
            time with parts of the the image border removed from each side

    Return:
        An instance of ImageSearch() with image_list added to it with the
            matcher trained on them
    """
    image_database = ImageSearch(lowes_ratio=ratio)

    crop_info = None
    if enriched_db:
        crop_info = CropImageInfo(0.15, 0.08, 0.25, 0.2)

    for image_info in image_list:
        image_database.add_image(image_info)
        image_database.add_image(image_info, crop_info=crop_info)

    image_database.matcher.train()

    return image_database
