from fastcore.script import *
from ultralytics import YOLO
import torch
import os
import wandb

@call_parse
def train_model(base_model:str, # Base model to use. Either yolov8n, yolov8s, yolov8m, yolov8l or yolov8x
                data_path:str, # Path to data folder
                project_dir:str, # Path to the project directory
                dataset:str, # Dataset to use, either 'hp', 'spk' or 'both'
                outdir:str='runs', # Where to save the model, default 'runs'
                epochs:int=50, # How many epochs to train 
                patience:int=30, # Epochs to wait for no observable improvement for early stopping
                batch:int=-1, # Batch size to use
                imgsz:int=640, # Image size to use
                optimizer:str='SGD', # Optimizer to use, one of [SGD, Adam, AdamW, RMSProp]
                cos_lr:bool=False, # Whether to use cosine annealing
                ):
    if dataset not in ['hp', 'spk', 'both']:
        print('Invalid dataset, stopping')
        return
    if base_model not in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']:
        print('Invalid base model, defaulting to yolov8s')
        base_model = 'yolov8s'

    if optimizer not in ['SGD', 'Adam', 'AdamW', 'RMSProp']:
        print('Invalid optimizer, defaulting to SGD')
        optimizer = 'SGD'

    optimizer_params = {
        'SGD': {
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005
        },
        'Adam': {
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0
        },
        'AdamW': {
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-3
        },
        'RMSProp': {
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.0,
            'weight_decay': 0.0
        }
    }

    opt = optimizer_params[optimizer]

    print(f'Starting to train with parameters {base_model} {optimizer} {data_path}')

    datasets = {
        'hp': 'hiidenportti_dw.yaml',
        'spk': 'spk_dw.yaml',
        'both': 'both.yaml'
    }

    wandb.init(project='yolov8_deadwood', job_type='training', name=f'{base_model}_{dataset}')
    model = YOLO(f'{project_dir}/yolo_models/{base_model}-seg.pt')
    results = model.train(data=f'{data_path}{datasets[dataset]}',
                          epochs=epochs,
                          patience=patience,
                          imgsz=imgsz,
                          batch=batch,
                          project=f'{project_dir}/{outdir}',
                          name=f'{base_model}_{optimizer}_{dataset}',
                          cache='ram',
                          exist_ok=True,
                          optimizer=optimizer,
                          cos_lr=cos_lr,
                          lr0=opt['lr0'],
                          lrf=opt['lrf'],
                          momentum=opt['momentum'],
                          weight_decay=opt['weight_decay'],
                          device=0,
                          scale=0.25,
                          flipud=0.5)

    wandb.finish()
