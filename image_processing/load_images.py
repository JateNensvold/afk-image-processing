
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import image_processing.globals as GV
import image_processing.database.imageDB as imageSearchDB
import image_processing.processing as processing
import collections
import math
import matplotlib
import image_processing.afk.hero_object as HO


def display_image(image, multiple=False, display=GV.DEBUG, color_correct=True,
                  colormap=None):
    backend = matplotlib.get_backend()

    if backend.lower() != 'tkagg':
        if GV.VERBOSE_LEVEL >= 1:
            print("Backend: {}".format(backend))
        plt.switch_backend("tkagg")

    if not display:
        return

    if multiple:
        image = concat_resize(image)
    elif color_correct and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.ion()

    plt.figure()
    if colormap:
        plt.imshow(image, cmap="gray")

    else:
        plt.imshow(image)

    plt.show()
    input('Press any key to continue...')
    plt.close("all")


def findFiles(path: str, flag=True, lower=False):
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


def clean_hero(img: np.array, lower_boundary: int, upper_boundary: int,
               size, img_contour):
    EDGE_TOLERANCE = 0.1
    width = abs(size[0] - size[1])
    height = abs(size[2] - size[3])
    pixel_tolerance = ((width + height) / 2) * EDGE_TOLERANCE
    # convert to RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    binary = cv2.inRange(gray, lower_boundary, upper_boundary, 255)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    canvas = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = np.zeros_like(img)
    for index, component in enumerate(zip(contours, hierarchy)):
        currentContour = component[0]
        # currentHierarchy = component[1]

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

            cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)
            cv2.drawContours(canvas, [approx], -1,
                             (0, 0, 0), 1, cv2.LINE_AA)
            point = (inner_x, inner_y)
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

    return mask


def colorClassify(img: np.ndarray, contour):
    """

    Args:

    Return:

    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canvas = img.copy()
    counter = 0
    center = [i/2 for i in img.shape]
    lower_x = img.shape[0]
    upper_x = 0
    lower_y = img.shape[1]
    upper_y = 0
    colors = []
    for i in range(len(contour) - 1):
        pt = contour[i]
        pt2 = contour[i+1]
        lower_x = min(pt[0][0], lower_x)
        upper_x = max(pt[0][0], upper_x)
        lower_y = min(pt[0][1], lower_y)
        upper_y = max(pt[0][1], upper_y)

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

                if tc_point[0] < gray.shape[0] and tc_point[1] < gray.shape[1]:
                    color = gray[tc_point[0]][tc_point[1]]
                    colors.append(color)

    size = (lower_x, upper_x, lower_y, upper_y)

    counter = collections.Counter(colors)
    common = counter.most_common()
    alpha_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = cv2.bitwise_not(np.zeros_like(alpha_img))
    boundary = 10
    for i in range(10):
        color = common[i][0]
        temp_mask = clean_hero(img, int(color - boundary),
                               int(color + boundary), size, contour)

        mask = cv2.bitwise_and(mask, cv2.bitwise_not(temp_mask))
    alpha_img = cv2.bitwise_and(alpha_img, mask)
    return alpha_img, mask


def crop_heroes(images: list, x_left=None, x_right=None, y_top=None,
                y_bottom=None, border_width=0.25):
    """
    Args:
        images: list of images to crop frame from
        border_width: percentage of border to take off of each side of the
            image
        removeBG: flag to attempt to remove the background from each hero
            returned(ensure this has not already been done to image earlier on
            in the process)
    Returns:
        dict of name as key and  images as values with 'border_width'
            removed from each side of the images
    """
    sides = {"x_left": x_left, "x_right": x_right,
             "y_top": y_top, "y_bottom": y_bottom}

    cropHeroes = []
    for _name, _side in sides.items():
        if _side is None:
            sides[_name] = border_width

    for image in images:
        shape = image.shape
        x = shape[0]
        y = shape[1]

        left = round(sides["x_left"] * x)
        right = round(sides["x_right"] * x)
        top = round(sides["y_top"] * y)
        bottom = round(sides["y_bottom"] * y)

        crop_img = image[top: x-bottom, left: y-right]
        cropHeroes.append(crop_img)

    return cropHeroes


def concat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    # w_max = max(img.shape[1]
    #             for img in img_list)
    h_max = max(img.shape[0]
                for img in img_list)
    # resizing images
    im_list_resize = []
    for img in img_list:
        height, width = img.shape[:2]
        # ratio = width/height
        # width_dist = w_max - int(width/ratio)
        # height_dist = h_max - height

        scale = h_max/height
        resize = cv2.resize(img, (int(width * scale), int(height * scale)),
                            interpolation=interpolation)
        # if width_dist < height_dist:
        #     resize = cv2.resize(img,
        #                         (w_max, min(h_max, int(height*w_max/width))),
        #                         interpolation=interpolation)
        # else:
        #     resize = cv2.resize(img,
        #                         (min(w_max, int(width*h_max/height)), h_max),
        #                         interpolation=interpolation)

        new_h, new_w = resize.shape[:2]

        canvas = np.zeros((h_max, new_w, 3))
        shapeSize = len(resize.shape)
        if len(resize.shape) == 2:
            resize = cv2.merge([resize, resize, resize])

        if shapeSize < 3:
            canvas[0:new_h, 0:new_w] = resize
        else:
            canvas[0:new_h, 0:new_w, 0:3] = resize

        im_list_resize.append(canvas)

    return np.hstack(im_list_resize).astype(np.uint8)


def build_flann(image_list: list[HO.hero_object], ratio=0.8):
    """
    Build database of heroes to match to

    Args:
        image_list: list of (image(np.array), name(str), faction(str)) tuples
            to build database from

    Return:
        An instance of imageSearch() with image_list added to it with the
            matcher trained on them
    """
    imageDB = imageSearchDB.imageSearch(lowesRatio=ratio)
    for _Image_info_index, image_info in enumerate(image_list):

        imageDB.add_image(image_info, image_info.image)
    imageDB.matcher.train()

    return imageDB


if __name__ == "__main__":

    # Load in base truth/reference images
    # files = findFiles("../heroes/*jpg")
    # baseImages = []
    # for i in files:
    #     hero = cv2.imread(i)
    #     name = os.path.basename(i)
    #     baseImages.append((name, hero))

    # imageDB = build_flann(baseImages)
    import image_processing.build_db as build
    imageDB = build.loadDB()

    # load in screenshot of heroes
    image = cv2.imread("../test_ss.png")
    plt.imshow(image)
    plt.show()
    heroesDict = processing.getHeroes(image)

    cropHeroes = crop_heroes(heroesDict)

    for k, v in cropHeroes.items():
        hero_info, baseHeroImage = imageDB.search(v)
        x = heroesDict[k]["dimensions"]["x"]
        y = heroesDict[k]["dimensions"]["y"]

        coords = (x[0], y[0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(
            image, hero_info.name, coords, font, fontScale, color, thickness,
            cv2.LINE_AA)

    plt.figure()
    plt.imshow(image)
    plt.show()
    cv2.imwrite("labeledImage.png", image)
