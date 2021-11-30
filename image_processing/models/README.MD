# Training Models

This directory is used to store pytorch/detectron2/yolov5 models after they have been created

## Training FI_SI_STARS model with yolov5
https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208

Download training material from ROBOFLOW
https://app.roboflow.com/nate-jensvold/afk-arena-dataset/2
```bash
python train.py --img 416 --batch 16 --epochs 1000 --data si_fi_stars/data.yaml --cfg ./si_fi_stars/custom_yolov5s.yaml --weights '' --name si_fi_stars_model --cache
```

## Training Ascension Model with detectron2
