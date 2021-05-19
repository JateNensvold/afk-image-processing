import os

import pickle
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import image_processing
import math

import image_processing.processing as processing


def findFiles(path: str, flag=False, lower=False):
    """
    Finds and returns all filepaths that match the pattern/path passed
        in under 'path'

    Args:
        path: glob/regex to path of file(s)
        flag: flag to force file loading if it contains "_" or "-"
        lower: flag to load all filepaths as lowercase
    Return:
        Sorted list of file paths
    """
    valid_images = []
    images = glob.glob(path)
    # images = os.listdir(path)
    for i in images:
        if flag or ("_" not in i and "-"not in i):
            if lower:
                i = i.lower()
            valid_images.append(i)
    return sorted(valid_images)


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def clean_hero(img: np.array, lowerb: int, upperb: int, size, img_contour):
    EDGE_TOLERANCE = 0.1
    # print("lower: {} upper: {} size: {}".format(lowerb, upperb, size))
    width = abs(size[0] - size[1])
    height = abs(size[2] - size[3])
    pixel_tolerance = ((width + height) / 2) * EDGE_TOLERANCE
    # print("Pixel: {}".format(pixel_tolerance))
    # convert to RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # gray = processing.blur_image(gray)

    binary = cv2.inRange(gray, lowerb, upperb, 255)
    # plt.imshow(binary)
    # plt.show()
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # contours, hierarchy = cv2.findContours(
    #     edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get the actual inner list of hierarchy descriptions
    # hierarchy = hierarchy[0]
    canvas = img.copy()
    # print(contours)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = np.zeros_like(img)
    for index, component in enumerate(zip(contours, hierarchy)):
        currentContour = component[0]
        currentHierarchy = component[1]

        arclen = cv2.arcLength(currentContour, True)
        eps = 0.0005
        epsilon = arclen * eps
        approx = cv2.approxPolyDP(currentContour, epsilon, True)
        lowest = 0
        inner_x = approx[0][0][0]
        inner_y = approx[0][0][1]
        l_point = img.shape[1]
        r_point = 0
        t_point = 0
        b_point = img.shape[0]
        for pt in approx:
            x, y = pt[0][0], pt[0][1]
            distance = cv2.pointPolygonTest(
                img_contour, (int(x), int(y)), True)
            if distance > lowest:
                lowest = distance
                inner_x = x
                inner_y = y
            if x < l_point:
                l_point = x
            if x > r_point:
                r_point = x
            if y > t_point:
                t_point = y
            if y < b_point:
                b_point = y
        if lowest < pixel_tolerance:
            # print("distance: {}".format(lowest))

            # print((x, y), distance)
            cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)
            cv2.drawContours(canvas, [approx], -1,
                             (0, 0, 0), 1, cv2.LINE_AA)
            point = (inner_x, inner_y)

            # corners = {"b_left": (size[0], size[2]),
            #            "b_right": (size[0], size[3]),
            #            "t_left": (size[1], size[2]),
            #            "t_right": (size[1], size[3])}
            corners = {"b_left": (0, 0),
                       "b_right": (0, img.shape[1]),
                       "t_left": (img.shape[1], 0),
                       "t_right": (img.shape[1], img.shape[1])}
            distance = {}
            for k, v in corners.items():
                d = math.hypot((v[0] - inner_x), abs(inner_y - v[1]))

                distance[d] = (k, v)
            closest_corner = sorted(distance.keys())[0]
            corner = distance[closest_corner]

            color = (255,)*img.shape[2]

            cv2.rectangle(mask, corner[1], point, color, -1)
    # plt.imshow(canvas)
    # plt.show()

    return mask


def colorClassify(img: np.ndarray, contour):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # processing.blur_image(gray)

    canvas = img.copy()
    counter = 0
    center = [i/2 for i in img.shape]
    print("center:", center)
    lower_x = img.shape[0]
    upper_x = 0
    lower_y = img.shape[1]
    upper_y = 0
    print("shape", img.shape)
    # points = set()
    colors = []
    for i in range(len(contour) - 1):
        pt = contour[i]
        pt2 = contour[i+1]
        lower_x = min(pt[0][0], lower_x)
        upper_x = max(pt[0][0], upper_x)
        lower_y = min(pt[0][1], lower_y)
        upper_y = max(pt[0][1], upper_y)
        # counter += 1

        # dot_x = pt[0][0]
        # dot_y = pt[0][1]
        # print("dot", dot_x, dot_y, gray[dot_x][dot_y])
        cv2.circle(canvas, (pt[0][0], pt[0][1]), 2, (0, 255, 0), -1)
        cv2.drawContours(canvas, [contour], -1,
                         (0, 0, 255), 1, cv2.LINE_AA)

        from skimage.draw import line
        # being start and end two points (x1,y1), (x2,y2)
        start = [pt[0][0], pt[0][1]]
        end = [pt2[0][0], pt2[0][1]]
        discrete_line = list(zip(*line(*start, *end)))
        for point in discrete_line:
            towardsCenter = list(zip(*line(
                point[0], point[1], int(center[0]), int(center[1]))))
            for j in range(6):
                tc_point = towardsCenter[j]
                # print("tc", tc_point, type(tc_point))

                if tc_point[0] < gray.shape[0] and tc_point[1] < gray.shape[1]:
                    color = gray[tc_point[0]][tc_point[1]]
                    colors.append(color)

        # print(discrete_line)
        # if counter > 0:
        #     break
    size = (lower_x, upper_x, lower_y, upper_y)

    # print("size x", lower_x, upper_x, upper_x-lower_x)
    # print("size y", lower_y, upper_y, upper_y-lower_y)
    # histr, _ = np.histogram(colors, len(points))
    import collections
    counter = collections.Counter(colors)
    common = counter.most_common()
    alpha_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = cv2.bitwise_not(np.zeros_like(alpha_img))
    boundary = 10
    for i in range(10):
        color = common[i][0]
        # print(color, type(color), i)
        temp_mask = clean_hero(img, int(color - boundary),
                               int(color + boundary), size, contour)

        mask = cv2.bitwise_and(mask, cv2.bitwise_not(temp_mask))
    alpha_img = cv2.bitwise_and(alpha_img, mask)
    return alpha_img, mask


def crop_heroes(images: dict, border_width=0.25):
    """
    Args:
        images: dictionary with key value pairs as
            imageName: [image, imageOutline]
        border_width: percentage of border to take off of each side of the
            image
    Returns:
        dict of name as key and  images as values with 'border_width'
            removed from each side of the images
    """

    cropHeroes = {}

    for name, imageData in images.items():
        image = imageData["image"]
        poly = imageData["poly"]
        mask_output = colorClassify(image, poly)
        output = mask_output[0]
        shape = output.shape
        x = shape[0]
        y = shape[1]
        x_30 = round(x * border_width)
        y_30 = round(y * border_width)
        print("({},{})x({},{})".format(x_30, x-x_30, y_30, y-y_30))
        crop_img = output[y_30: y-y_30, x_30: x-x_30]
        cropHeroes[name] = crop_img

    return cropHeroes


# def match_heroes(baseHeroes: list, unknownHero, lowesRatio: int = 0.8) -> list:
#     """
#     Args:
#         baseHeroes: list of heroes to search for matches (base truth)
#         unknownHero: hero to attempt to identify
#         lowesRatio: ratio to apply to all feature matches(0.0 - 1.0), higher
#             is less unique (https://stackoverflow.com/questions/51197091/
#                 how-does-the-lowes-ratio-test-work)
#     Return:
#         list of baseHeroes index that had the highest number of "good" matches,
#             (aka matches that passed the lowe's ratio test)
#     """
#     sift = cv2.SIFT_create()

#     kp1, des1 = sift.detectAndCompute(unknownHero, None)
#     FLANN_INDEX_KDTREE = 0

#     hero_matches = {}
#     for index, hero in enumerate(baseHeroes):
#         kp2, des2 = sift.detectAndCompute(hero, None)

#         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=50)

#         flann = cv2.FlannBasedMatcher(index_params, search_params)

#         matches = flann.knnMatch(des1, des2, k=2)
#         good = []
#         good_parameter = 0.8
#         for m, n in matches:
#             if m.distance < good_parameter*n.distance:

#                 good.append(m)
#         if len(good) not in hero_matches:
#             hero_matches[len(good)] = {}
#         hero_matches[len(good)][index] = {}
#         hero_matches[len(good)][index]["good"] = good
#         hero_matches[len(good)][index]["hero"] = hero
#         hero_matches[len(good)][index]["matches"] = matches

#     matches = hero_matches.keys()
#     matches = sorted(matches, reverse=True)
#     best = matches[0]

#     for good_match_num in matches:
#         print(good_match_num, "matched heroes:", len(
#             hero_matches[good_match_num]),
#             [matchedHero for matchedHero in hero_matches[good_match_num]])
#     return [matchedHero for matchedHero in hero_matches[best]]

def get_good_features(matches: list, imagelist: list, ratio: int):
    good_features = []
    # good_parameter = 0.8
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_features.append(m)

    return good_features


class imageSearch():
    """
    Wrapper around an in memory image database

    Uses flann index to store and search for images base on extracted
        SIFT features

    Args:
        lowesRatio: ratio to apply to all feature matches(0.0 - 1.0), higher
            is less unique (https://stackoverflow.com/questions/51197091/
                how-does-the-lowes-ratio-test-work)
    """

    def __init__(self, lowesRatio: int = 0.8):
        self.ratio = lowesRatio
        # FLANN_INDEX_KDTREE = 1
        # index_param = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        # search_param = {"checks": 50}
        # self.flann = cv2.FlannBasedMatcher(index_param, search_param)
        self.matcher = cv2.BFMatcher(cv2.NORM_L1)
        self.extractor = cv2.SIFT_create()
        # self.extractor = cv2.ORB_create()\
        self.images = []

    def add_image(self,  name: str, hero: np.array):
        """
        Adds image features to image database and stores image
            in internal list for later use

        Args:
            name: name to return on match to image
            hero: image to add to database
        Return:
            None
        """
        # sift = cv2.SIFT_create()
        kp, des = self.extractor.detectAndCompute(hero, None)

        self.matcher.add([des])

        self.images.append((name, hero))

    def search(self, hero: np.array, display: bool = False,
               min_features: int = 5):
        """
        Find closest matching image in the database

        Args:
            hero: cv2 image to find a match for
            display: flag to show base image and search image for humans to
                compare/confirm that the search was successful
        Returns:
            Tuple containing (name, image) of the closest matching image in
                the database
        """
        # sift = cv2.SIFT_create()

        kp, des = self.extractor.detectAndCompute(hero, None)

        # matches = self.matcher.knnMatch(des, np.zeros_like(des), k=1)
        # matches = self.matcher.match(np.array(des))
        matches = self.matcher.knnMatch(np.array(des), k=2)

        # good_features = []
        # # good_parameter = 0.8
        # for m, n in matches:
        #     if m.distance < self.ratio * n.distance:
        #         good_features.append(m)

        # good_features = sorted(good_features, key=lambda x: x.distance)
        ratio = self.ratio
        good_features = []
        while len(good_features) < 5 and ratio <= 1.0:
            good_features = get_good_features(matches, self.images, ratio)
            ratio += 0.05
        if len(good_features) < 5:
            good_features = matches[0]

        matches_by_id = [[] for _ in range(len(self.images))]
        for matchIndex in range(len(good_features)):
            match = good_features[matchIndex]
            m = match
            matches_by_id[m.imgIdx].append(m)

        good_features = sorted(good_features, key=lambda x: x.distance)

        matches_idx = [x.imgIdx for x in good_features]
        import collections
        counter = collections.Counter(matches_idx)
        common = counter.most_common()
        itr = 0
        matchLen = len(good_features)
        while itr < 5 and itr < len(common):
            commonEntry = common[itr]
            commmonIndex = commonEntry[0]
            commonOccurances = commonEntry[1]
            heroName, heroImage = self.images[commmonIndex]
            print("Name: {} Votes: {}/{} ({})".format(
                heroName, commonOccurances, matchLen,
                commonOccurances/matchLen))
            itr += 1

        name, baseHero = self.images[common[0][0]]
        # plt.figure()
        # plt.imshow(hero)
        # plt.figure()
        # plt.imshow(baseHero)
        # plt.show()

        def concat_resize(img_list, interpolation=cv2.INTER_CUBIC):
            # take minimum width
            w_max = max(img.shape[1]
                        for img in img_list)
            h_max = max(img.shape[0]
                        for img in img_list)
            # resizing images
            im_list_resize = [cv2.resize(img,
                                         (w_max, h_max),
                                         interpolation=interpolation)
                              for img in img_list]
            # return final image
            for i in im_list_resize:
                print(i.shape)
            return np.concatenate((im_list_resize[0][:, :, 2],
                                   im_list_resize[1][:, :, 2]),
                                  axis=1)

        # function calling
        if display:
            catHeroes = concat_resize([baseHero, hero])

            plt.figure()
            plt.imshow(catHeroes)
            plt.show()
        return name, baseHero


def build_flann(baseHeroes: list):
    """
    Build database of heroes to match to

    Args:
        baseHeroes: list of (hero:name) tuples to build database from

    Return:
        An instance of imageSearch() with baseHeroes added to it with the
            matcher trained on them
    """
    imageDB = imageSearch(lowesRatio=0.8)
    for index, heroTuple in enumerate(baseHeroes):
        name = heroTuple[0]
        hero = heroTuple[1]
        imageDB.add_image(name, hero)
    imageDB.matcher.train()

    return imageDB


if __name__ == "__main__":

    # Load in base truth/reference images
    files = findFiles("../heroes/*jpg")
    baseImages = []
    for i in files:
        hero = cv2.imread(i)
        name = os.path.basename(i)
        baseImages.append((name, hero))

    # load in screenshot of heroes
    image = cv2.imread("../test_ss.png")
    plt.imshow(image)
    plt.show()
    heroesDict = processing.getHeroes(image)

    # heroes[name]["out"]=out
    # heroes[name]["poly"]=poly
    # heroes[name]["dimensions"]={}
    # heroes[name]["dimensions"]["x"]=(d[3], d[3]+d[0])
    # heroes[name]["dimensions"]["y"]=(d[2], d[2]+d[1])

    cropHeroes = crop_heroes(heroesDict)

    imageDB = build_flann(baseImages)
    
    for k, v in cropHeroes.items():
        name, baseHeroImage = imageDB.search(v)
        # print(heroesDict.keys())
        x = heroesDict[k]["dimensions"]["x"]
        y = heroesDict[k]["dimensions"]["y"]

        coords = (x[0], y[0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(
            image, name, coords, font, fontScale, color, thickness,
            cv2.LINE_AA)

    plt.figure()
    plt.imshow(image)
    plt.show()
    cv2.imwrite("labeledImage.png", image)
