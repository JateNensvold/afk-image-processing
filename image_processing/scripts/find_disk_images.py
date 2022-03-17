import os
import argparse
import re
import shutil


def copy_file(source_file_path: str, destination_folder: str):
    """_summary_

    Args:
        source_file_path (str): _description_
        destination_folder (str): _description_
    """
    print(f"Copying file from '{source_file_path}' to '{destination_folder}'")
    destination_file = os.path.join(
        destination_folder, os.path.basename(source_file_path))
    shutil.copyfile(source_file_path, destination_file)


def recursive_find(folder_path: str, destination: str, regex_pattern: str, ):
    """_summary_

    Args:
        folder_path (str): _description_
        destination (str): _description_
        regex_pattern (str): _description_
    """
    folder_files = os.listdir(folder_path)
    for file_name in folder_files:
        new_path = os.path.join(folder_path, file_name)
        if os.path.isdir(new_path):
            recursive_find(new_path, destination, regex_pattern)
        else:
            regex_match = re.findall(
                regex_pattern, new_path, re.IGNORECASE | re.MULTILINE)
            # print(regex_pattern, regex_match, new_path)
            if regex_match:
                copy_file(new_path, destination)


def main():
    """_summary_
    """

    parser = argparse.ArgumentParser(
        description="Recursively find all images that match a regex and move "
        "them to a destination folder")

    parser.add_argument("start_folder", type=str,
                        help="Folder to start recursive search from")
    parser.add_argument("destination_folder", type=str,
                        help="Location to put all files that match glob")
    parser.add_argument("image_glob", type=str,
                        help="Glob to match images with")

    args = parser.parse_args()

    # output =  re.findall(args.image_glob, "../database/hero_icon/Mauler/Safiya/NOD_Battle_5101.png", re.IGNORECASE | re.MULTILINE)
    # print(output)
    recursive_find(args.start_folder, args.destination_folder, args.image_glob)


if __name__ == "__main__":
    main()
