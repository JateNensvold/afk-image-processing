from enum import Enum
import os
import re
from pathlib import Path

from collections import defaultdict
import subprocess


class RequiredType(Enum):
    required: str = "required"
    optional: str = "optional"


def rename_heroes(directory: Path):
    """
    Rename any heroes without optional/required in their name to be a
    required hero

    Args:
        directory (Path): _description_
    """

    hero_list = os.listdir(directory)

    hero_dict: dict[str, list[Path]] = defaultdict(lambda: defaultdict(list))

    for image_index, image_name in enumerate(hero_list):
        split_list = re.split(r"\-|_|\.", image_name)
        split_name = split_list[0]
        # if split_name in hero_dict:
        old_path = Path(directory, image_name)

        answer = None
        while answer not in ["y", "n"]:
            output = subprocess.run(
                ["code", str(old_path)], shell=False, text=True, check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            answer = input(f"Is {old_path} required? y/n ")
        if answer == "y":
            required_type = RequiredType.required
        else:
            required_type = RequiredType.optional

        hero_index = len(hero_dict[split_name][required_type])
        if required_type == RequiredType.required:
            hero_index += 1
        portrait_name = (
            f"{split_name}-{required_type.name}{hero_index}"
            f"{old_path.suffix}").lower()
        portrait_path = Path(directory, portrait_name)
        old_path.rename(portrait_path)
        print(f"Renamed Image #{image_index} - {old_path} -> "
              f"{portrait_path.name}")
        hero_dict[split_name][required_type].append(portrait_name)


if __name__ == "__main__":
    local_dir = Path(__file__).parent
    rename_heroes(local_dir.joinpath("heroes"))
