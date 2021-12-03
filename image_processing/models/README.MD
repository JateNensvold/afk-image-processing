# Training Models

This directory is used to store pytorch/detectron2/yolov5 models after they have been created

## Training FI_SI_STARS model with yolov5
https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208

Download training material from ROBOFLOW
https://app.roboflow.com/nate-jensvold/afk-arena-dataset/3

Unzip data into si_fi_stars folder in models directory
```bash
cd .../afk-image-processing/image_processing/models/si_fi_stars
curl -L <curl_string> > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```
Open up si_fi_stars/data.yaml
Edit the following lines
```yaml
train: ../train/images
val: ../valid/images
```
to
```yaml
train: ./train/images
val: ./valid/images
```

Start Training
Executing the following command inside the si_fi_stars directory
```bash
python ../yolov5/train.py --img 416 --batch 16 --epochs 1000 --data ./data.yaml --cfg ./custom_yolov5s.yaml --weights '' --name si_fi_stars_model --cache
```

After Training
Move the model from `yolov5/runs/train/si_fi_stars_model/weights/` to `afk_processing/image_processing/models/final_models`
and rename to `fi_start_model.pt`

## Training Ascension Model with detectron2
