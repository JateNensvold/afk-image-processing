# AFK-Imagine-Processing
afk_image_processing is a feature recognition application for the game AFK-Arena. It makes use of OpenCV, Faster RCNN and Yolov5 to
detect in game characters and a facet of features that each character can posses by processing game screenshots.

This repository can detect character names, FI level, SI level and ascension.

## Environment
When initializing the environment a build architecture needs to be selected.
Supported build options are `CUDA`(GPU support) and `CPU` which will setup the environment
for the selected architecture type

To set the build type go to
[.devcontainer/docker-compose.yml](.devcontainer/docker-compose.yml) and set
BUILD_TYPE to the desired architecture

```
services:
  afk_processing_container:
    build:
    ...
      args:
        BUILD_TYPE: "CUDA"
    ...
```


## Deploying Application with albedo-bot

### Dependencies
- docker
- docker-compose
- tmux
- git-lfs




1. Project Download
Create a directory in your desired file location called `projects`
with the command 

```bash
cd <host-dir> && mkdir projects
```

2. Checkout code to deploy
```bash
git checkout git@github.com:JateNensvold/afk_image_processing.git
git checkout git@github.com:JateNensvold/albedo-bot.git
```
Go to `projects/afk_image_processing/.devcontainer/docker-compose.yaml` and 
ensure that the `BUILD_TYPE` is set to the proper argument for your production hardware(CPU or CUDA)

3. Build docker image

Move to `afk_image_processing` git repo
```bash
cd projects/afk_image_processing
```
Go to devcontainer folder
```
cd .devcontainer
```
Build docker images
```
docker compose build
```
4. Start the docker containers

```bash
docker compose up -d
```

5. Connect to dev container
```bash
docker exec -it devcontainer-afk_processing_container-1 bash
```

6. Install package dependencies

Go to post-install command in `.devcontainer/devcontainer.json` for the latest command
the following may be out of date
```bash
cd /workspace/albedo-bot && pip3 install -r requirements.txt  && /workspace/afk_image_processing/requirements/install ${BUILD_TYPE}
```

7. Run database creation
```bash
# Add albedo-bot token listed in the discord api developer portal to
# /workspace/albedo_bot/albedo_bot/config/config.py in the `token` field,
# or follow the commands printed on the screen from running one of the commands below


# To create a new database
cd /workspace/albedo-bot && python3 albedo_bot/albedo_main.py init
# To reinitialize a database
cd /workspace/albedo-bot && python3 albedo_bot/albedo_main.py reset
```

```
cd /workspace/afk_image_processing && python3 image_processing/build_db.py
```

8. Run image-processing service
Disconnect from docker session and start a tmux session to run the
image-processing service in
```bash
tmux new
```
Connect to docker container
```bash
docker exec -it devcontainer-afk_processing_container-1 bash
```
Run command to start processing server
```
cd /workspace/afk_image_processing/ && python3 image_processing/processing_server.py
```
Disconnect from tmux session by hitting `ctrl + 'b'` pause `d`

9. Run albedo-bot service
Create a second tmux session
```bash
tmux new
```
Connect to container again
```bash
docker exec -it devcontainer-afk_processing_container-1 bash
```
```bash
cd /workspace/albedo-bot/ && python3 albedo_bot/albedo_main.py
```

10. Modifying Bot

Bot should be up and running, to restart or view the console of either application 
run `tmux ls` and then `tmux attach -t <window number>` to attach to your process window



## Setup For development
When developing or executing the afk_image_processing codebase dependencies and
environment setup are required before the code can be ran.

python3 image_processing/models/yolov5/train.py --img 416 --batch 16 --epochs 3000 --data image_processing/models/training_data/yolo_data/yolo_data/data.yaml --weights yolov5s.pt

1. Initialize submodules
This repository makes use of a git submodules called YoloV5 for training some
of the image recognition Neural Networks. To initialize that submodule for the
first time after checking out the `afk_image_processing` the
following commands needs to be ran
```bash
git submodule init image_processing/models/yolov5/
git submodule update
```
To 
"image_processing/afk/fi/fi_detection/yolov5" submodule init


## Training new models
### Training new Hero Ascension/Border Model for Detectron

1. Download dataset

To download the JSON file provided by labelbox with the pretrained data go to the following link
[labelbox.json](https://storage.googleapis.com/labelbox-exports/ckrz9srr563510ydl0bpv94w1/ckrzarwja0y510y9u4dc009id/export-2022-03-12T08%3A28%3A41.743Z.json?GoogleAccessId=api-prod%40labelbox-193903.iam.gserviceaccount.com&Expires=1648283347&Signature=QUM8%2B0F6yY2jYNF%2BtcjNLF6j5lUPC8%2BO%2BwLNRj%2F2%2BPQTTKcnaFtwTYoQSKRYXaF3fCmOuDLA4ywMVvIifFjvbBrkJMYOTD5tVq5f6hc9zjYpMi2xa3fYA7sfIOTwmX2kjJwAXICIghsGB0yQBp%2FGbstkIejegM3bk43gvTBVRZ%2BZK9R3TTdQ4G1sUqT6BcAm88n6H3eq4XOALYnXtMLohlLnJeMGy8p0M5v%2F1y841dtQ14pgrbHhJ%2FtbeTKd%2BfyVfmb5R%2FPxzeBs%2BR7l60ZppapM7whQWrIaUS9i8GCN3GMp5kIDk4wAeElZKOxMWHK5dew9z2%2Fefw4P%2B%2BWgQF7UKQ%3D%3D&response-content-disposition=attachment)

2. Move the dataset

The dataset has to be relocated to the following location `./image_processing/models/training_data/coco_data`
so we can run a special conversion script that will convert the data from the
`labelbox` monolith file into the COCO format 

3. Convert Data

Run the following command from the `./image_processing/models/training_data/coco_data`
directory to convert the files to COCO format, dont forget to replace `<name of json file>`
```bash
python3 ../../../scripts/box_label_to_coco.py <name of json file> ./
```
After running the above command a folder will exist that is named `coco_data`
followed by a number. This will be the data getting passed to detectron2 to
train a model

4. Train Model

From the `./image_processing/models/training_data/coco_data` directory run the
following command replacing `<coco_data_name>` with the name of the folder that
was created in the previous step
```bash
python3 detectron_train.py coco_data0/ ../../training_models/ascension_model/
```
After the above command finishes running the final model will be located at 
`./image_processing/models/training_models/ascension_model/model_final.pth`


** *Optional*  
5. Use model for image recognition  
To use the newly trained model with the afk_image_processing library replace
the old model located at `./image_processing/models/final_models/hero_ascension_model.pt`
with the model created in the previous step

### YoloV5 i.e. ascension/stars, fi, si

1. Download dataset  
Get afk-arena si, fi, and stars dataset from https://app.roboflow.com/nate-jensvold/afk-arena-dataset/2
and choose Export -> Format: Yolo V5 PyTorch -> Terminal while in the following folder
`./image_processing/models/training_data/si_fi_stars`

Ex. How to Download/unzip
```bash
# Replace the words <curl link> with the link/url generated by roboflow

curl -L "<curl link>" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

2. Fix Folder Paths
Due to an incompatibility issue between YoloV5 and roboflow data layout we need
to make a quick change to [data.yaml](./image_processing/models/training_data/si_fi_stars/data.yaml)
so that the values that are generated have the correct path between the yoloV5
repo and the location we just put out data. After downloading our data from 
Roboflow the `data.yaml` file should have two entries at the top that look
similar to the following
```yaml
train: ../train/images
val: ../valid/images
```
and we need to change them to the following format
```yaml
train: ../training_data/si_fi_stars/train/images
val: ../training_data/si_fi_stars/valid/images
```

2. Create and train YoloV5 Model

Now that our training data is ready to be trained on run the following command
from the `./image_processing/models/training_data/si_fi_stars` directory.

*Warning, the following command can take up to several hours to run, reduce
epochs to reduce runtime at the risk of lowering model accuracy*
```bash
python3 ../../yolov5/train.py --img 416 --batch 16 --epochs 3000 --data data.yaml --weights yolov5s.pt
```
After the above command has finished it will output a message with the exact
location of the models that were generated that will be located somewhere like
the following `./image_processing/models/yolov5/runs/train/<exp_x>/weights`.
In that folder there will be two models that have been saved, the one we are
looking for is called `best.pt`

Ex.
```
Model Summary: 213 layers, 7039792 parameters, 0 gradients
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 1/1 [00:00<00:00, 28.35it/s]                                                                           
                 all          1          2      0.976          1      0.995      0.995
              1 star          1          1          1          1      0.995      0.995
                9 fi          1          1      0.952          1      0.995      0.995
Results saved to ../../yolov5/runs/train/exp22
```

** *Optional*  
3. Use model for image recognition  
To use the newly trained model with the afk_image_processing library replace
the old model located at `./image_processing/models/final_models/fi_si_star_model.pt`
with the model called `best.pt` created in the previous step.

### Setup GPU Support in docker WSL2
https://forums.developer.nvidia.com/t/guide-to-run-cuda-wsl-docker-with-latest-versions-21382-windows-build-470-14-nvidia/178365
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

Complete steps 1 and 2 to enable GPU development in VSCode/Docker Devcontainers on WSL2

1. Install GPU package in windows from the above link

2. Install nvidia docker runtime
```bash
sudo apt-get install -y nvidia-docker2      
```

Step 3 is completed automatically by the devcontainer. It can be ignored if GPU
setup was enabled when running devcontainer setup (i.e `CUDA` was the selected
build option when configuring the `docker-compose` file)
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
*This step is ran automatically in the devcontainer setup
```bash
./requirements/install ${BUILD_TYPE}
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
To rebuild the character recognition database from source images run the
following command 
Ex. 
``` python
python3 image_processing/build_db.py 
```
or the feature detection command with the -r flag enabled

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
"image_processing/afk/fi/fi_detection/yolov5" submodule init
tests/data/images
