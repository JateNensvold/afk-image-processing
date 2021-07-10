import cv2
import os

if __name__ == "__main__":
    folder_path = "./database/hero_validation"
    files = os.listdir(folder_path)

    sizes = []

    for _file in files:
        _file_name = os.path.join(folder_path, _file)
        _image = cv2.imread(_file_name)
        sizes.append((_image.shape, _file_name))

    sizes.sort(key=lambda x: x[0][0])
    print("min", sizes[5:10])
    print("max", sizes[-5:-1])
