# AFK-Imagine-Processing
Afk-image-processing is a feature recognition application for the game AFK-Arena. It makes use of OpenCV, Faster RCNN and Yolov5 to
detect in game characters and a facet of features that each character can posses by processing game screenshots.

This repository can detect character names, FI level, SI level and ascension.

## Environment
When initializing the environment a build architecture needs to be selected.
Supported build options are 'GPU' and 'CPU' which will setup the environment for the selected architecture type
To set the build type go to .devcontainer/devcontainer.json and set 'build:args:BUILD_TYPE' to the desired type



## Training new models
### Detectron i.e border dataset
Get dataset at https://app.roboflow.com/nate-jensvold/afk-arena-dataset/2
Place COCO dataset at `/workspaces/afk-image-processing/image_processing/afk/fi/fi_detection/data/coco_data`
Call `/workspaces/afk-image-processing/image_processing/afk/fi/fi_detection/data/detectron_train.py` 
with CUDA/GPU architecture.

### YoloV5 i.e. ascension/stars, fi, si
```python
python3 yolov5/train.py --img 416 --batch 16 --epochs 3000 --data si_fi_stars/data.yaml --weights yolov5s.pt
```

### Setup GPU Support in docker WSL2
https://forums.developer.nvidia.com/t/guide-to-run-cuda-wsl-docker-with-latest-versions-21382-windows-build-470-14-nvidia/178365
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

Step 3 is completed automatically by the devcontainer. Complete steps 1 and 2 to enable GPU development in VSCode/Docker Devcontainers on WSL2

1. Install GPU package in windows

2. Install nvidia docker runtime
```bash
sudo apt-get install -y nvidia-docker2      
```

3. Install Cuda
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get update

apt-get install -y cuda-toolkit-11-4
```


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
