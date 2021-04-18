from skimage import io
from skimage.color import rgb2gray, rgba2rgb
import numpy as np
import cv2
import matplotlib.pyplot as plt


def feature_detect(image):
    """
    Detect features in image.
    Args:
        image: cv2 image format
    Return:
        image with features drawn on
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(False)

    kp = fast.detect(gray_img, None)
    kp_img = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))
    return kp_img


def get_corrected_img(img1, img2):
    """
    Compare img1 with img2
    Args:
        img1: image to compare
        img2: image to train off
    """
    MIN_MATCHES = 20

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print(len(good_matches), MIN_MATCHES)
    # if len(good_matches) > MIN_MATCHES:
    #     src_points = np.float32(
    #         [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     dst_points = np.float32(
    #         [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    #     corrected_img = cv2.warpPerspective(
    #         img1, m, (img2.shape[1], img2.shape[0]))

    #     return corrected_img
    return img2


if __name__ == "__main__":

    e_pippa = cv2.imread("./pippa.jpg")
    o_pippa = cv2.imread("./heroes/pippa.jpg")
    o_lorsan = cv2.imread("./heroes/lorsan.jpg")
    e_lorsan = cv2.imread("./lorsan.jpg")
    img = get_corrected_img(e_pippa, o_pippa)
    img = get_corrected_img(e_lorsan, o_lorsan)


    # image1 = feature_detect(image1)
    # image2 = feature_detect(image2)

    # plt.imshow(image1)
    # plt.show()

    # plt.imshow(image2)
    # plt.show()

    # plt.imshow(img)
    # plt.show()

    # cv2.imshow('FAST', kp_img)
    # cv2.waitKey()
    # cv2.imwrite("ROI_temp.png", kp_img)
