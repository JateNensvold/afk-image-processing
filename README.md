# AFK-Imagine-Processing
Afk-image-processing is a feature recognition application for the game AFK-Arena. It makes use of OpenCV, Faster RCNN and Yolov5 to
detect in game characters and a facet of features that each character can posses by processing game screenshots.

This repository can detect character names, FI level, SI level and ascension.

## Environment
When initializing the environment a build architecture needs to be selected.
Supported build options are 'GPU' and 'CPU' which will setup the environment for the selected architecture type
To set the build type go to .devcontainer/devcontainer.json and set 'build:args:BUILD_TYPE' to the desired type

## Dependencies

```python
pip3 install -r requirements.txt
pip3 install -e .
```

## Feature Detection
### Detect feature
Run feature recognition by calling the following function from the image_processing directory
```python
python3 image_processing/afk/si/get_si.py -i <path to image>
```

### Verbose output
For verbose output to help with debugging two levels of verbosity are available
``` python
python3 image_processing/afk/si/get_si.py ... -v
```
To print general information
and 
``` python
python3 image_processing/afk/si/get_si.py ... -vv
```
To print more detailed details about feature recognition

### Rebuild database
To rebuild the character recognition database from source images run the feature detection command with the -r flag
Ex. 
``` python
python3 image_processing/afk/si/get_si.py ... -r
```
### Debug mode
To see feature detection at various checkpoints as well as verbose output run the program with -d
Ex. 
``` python
python3 image_processing/afk/si/get_si.py ... -d
```

## TODO
Add instructions for generating datasets or how to download them
"image_processing/database/hero_icon" local
"image_processing/database/stamina_templates" local
"image_processing/afk/fi/fi_detection/data/coco_data/" from labelbox and run conversion to COCO dataset
"image_processing/afk/fi/fi_detection/models/" local
"image_processing/afk/fi/fi_detection/yolov5" submodule init
"image_processing/afk/fi/fi_detection/data/fi_stars_data" local
tests/data/images
