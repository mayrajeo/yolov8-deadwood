# Deadwood detection from RGB UAV imagery with YOLOv8-seg models

## About

Code repository for the article **UAVs and deep learning can enhance deadwood mapping in boreal forests**, under preparation.

## Installation

First install GDAL to your system. If you use conda then installing rasterio is enough, but with pip use instructions from [https://pypi.org/project/GDAL/](https://pypi.org/project/GDAL/).

Then the required libraries for running the are pretty much [`ultralytics`](https://github.com/ultralytics/ultralytics), [`sahi`](https://github.com/obss/sahi/) and [`geo2ml`](https://github.com/mayrajeo/geo2ml), which can be installed via pip:

```bash
pip install ultralytics
pip install sahi
pip install git+git://github.com/mayrajeo/geo2ml.git
```

## Training data

In this work, we used UAV RGB Orthomosaics from Hiidenportti, Kuhmo, Eastern-Finland and Sudenpesänkangas, Evo, Southern-Finland. Hiidenportti dataset has a spatial resolution of around 4 cm, and Sudenpesänkangas dataset has a spatial resolution of 4.85 cm. Hiidenportti data contains 9 different UAV mosaics, and Sudenpesänkangas data is one single orthomosaic. From these data, we created virtual plots to use as a training and validation data for the models. From Hiidenportti, we constructed 33 scenes of varying sizes in such way that all 9m circular field plots present in the area were covered, and each field plot center had at least 45 meter distance to the edge of the scene. For Sudenpesänkangas, due to the area and orthomosaic being larger, we extracted 100 x 100 m plots in such way that each scene contains only one circular field plot. In total, Hiidenportti data contained 33 scenes that cover 71 field plots, and Sudenpesänkangas data contained 71 virtual plots.

These scenes were then split into 640x640 pixel chips using [scripts/make_data.py](scripts/make_data.py).

## Training

Each trained model is available on Hugging Face platform: [https://huggingface.co/mayrajeo/yolov8-deadwood](https://huggingface.co/mayrajeo/yolov8-deadwood). The training script can be found from [scripts/train_models.py](scripts/train_models.py).

## Inference

All `.pt` files can be directly used with `ultralytics` library:

```python
from ultralytics import YOLO

model = YOLO(<path_to_model>)
results = model(<input_image>)
```

[scripts/predict_image.py] can be used to do sliced prediction on a larger UAV mosaic. This script loads selected model and performs sliced predictions with [https://github.com/obss/sahi/](https://github.com/obss/sahi/) library and outputs the predictions as a georeferenced geojson or gpkg, depending on how many there are. Postprocessing is done by using GREEDYNMM with IOS threshold of 0.2 based on the predicted masks as many times that no polygons are merged. Implementation details can be found on [src/postprocess.py](src/postprocess.py).