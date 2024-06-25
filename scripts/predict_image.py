from fastcore.script import *

import rasterio as rio
import pandas as pd
import geopandas as gpd

from shapely.geometry import box, Polygon, MultiPolygon
from geo2ml.data.coordinates import *

from sahi.predict import get_sliced_prediction
from sahi.models.yolov8 import Yolov8DetectionModel

import time

import torch
from pathlib import Path 

import sys, os
sys.path.append('..')
from src.postprocess import batched_greedy_nmm

def georef_sahi_preds(preds, path_to_ref_img, result_type='bbox') -> gpd.GeoDataFrame:
    "Converts a list of `ObjectPredictions` to a geodataframe, georeferenced according to reference image"
    labels = [p.category.id for p in preds]

    if result_type == 'bbox': 
        polys = [box(*p.bbox.to_xyxy()) for p in preds]
    elif result_type == 'mask': 
        polys = []
        for p in preds:
            segmentation = p.mask.segmentation
            temp_polys = []
            for segm in segmentation:
                xy_coords = [(segm[i], segm[i+1]) for i in range(0, len(segm), 2)]
                xy_coords.append(xy_coords[-1])
                temp_polys.append(Polygon(xy_coords))
            # Check for parts that are completely within a larger polygon of the same segmentation
            not_contained_polys = []
            for p in temp_polys:
                num_within = sum([p.buffer(0).within(d.buffer(0)) for d in temp_polys])
                if num_within == 1: not_contained_polys.append(p)

            polys.append(MultiPolygon(not_contained_polys))
    else:
        print(f'Unknown result type {result_type}, defaulting to bbox')
        
        polys = [box(*p.bbox.to_xyxy()) for p in preds]
    scores = [p.score.value for p in preds]
    gdf = gpd.GeoDataFrame({'label':labels, 'geometry':polys, 'score':scores})
    tfmd_gdf = georegister_px_df(gdf, path_to_ref_img)
    return tfmd_gdf

@call_parse
def main(yolov8_weights:str, # Path to yolov8 model weights to use
         tile:Path, # Path to UAV to process
         outpath:Path, # Directory to save the results
         use_cuda:bool, # Whether to use cuda if it is available
         use_tta:bool, # Whether to use Test-time augmentation
         half:bool, # Whether to use half-precision 
         image_size:int=640, # Image size for YOLOv8 model
         slice_size:int=640, # Slice size to use with sahi
         overlap_ratio:float=0.2, # Slice overlap ratio
         conf_th:float=0.25 # Confidence threshold for predictions
    ):
    """
    Run YOLOv8 model with sahi for larger UAV image, georeference the predictions 
    and save them to `outpath`
    """

    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

    print(f'Using {device} for predictions...')
    if half: print(f'Using half precision for inference')
    # Initialize model
    det_model = Yolov8DetectionModel(model_path=yolov8_weights,
                                     device=device,
                                     confidence_threshold=conf_th,
                                     image_size=image_size)
    
    det_model.model.overrides.update({
        'augment': use_tta,
        'half': half,
        'conf': conf_th,
        'retina_masks': True
    })
    
    # Use postprocess_match_threshold of 1.0 to get basically unprocessed predictions
    # Merging is handled differently later
    sliced_pred_results = get_sliced_prediction(str(tile), 
                                                det_model, 
                                                slice_width=slice_size,
                                                slice_height=slice_size,
                                                overlap_height_ratio=overlap_ratio,
                                                overlap_width_ratio=overlap_ratio,
                                                perform_standard_pred=False,
                                                verbose=2,
                                                postprocess_type='IOS',
                                                postprocess_match_threshold=1.0) 
    
    
    tile_fn = tile.stem
    result_type = 'mask' if det_model.model.overrides['task'] == 'segment' else 'bbox'
    print('Starting to georeference predictions')
    georef_start = time.time()
    tfmd_gdf = georef_sahi_preds(preds=sliced_pred_results.object_prediction_list,
                                 path_to_ref_img=tile, result_type=result_type)
    print(f'Georeferencing done in {time.time() - georef_start} seconds')
    tfmd_gdf = tfmd_gdf.explode(ignore_index=True)
    n_polygons = len(tfmd_gdf)
    print(f'{n_polygons} polygons before GREEDYNMM')
    n_iterations = 1
    while(True): # Iterate until no more polygons to merge
        iter_start = time.time()
        tfmd_gdf = batched_greedy_nmm(tfmd_gdf, 'IOS', 0.2)
        if len(tfmd_gdf) == n_polygons:
            print(f'No more changes after {n_iterations} iterations of GREEDYNMM')
            print(f'Last iteration took {time.time() - iter_start} seconds')
            break
        n_polygons = len(tfmd_gdf)
        print(f'{n_polygons} polygons after {n_iterations} iterations of GREEDYNMM')
        print(f'Last iteration took {time.time() - iter_start} seconds')
        n_iterations += 1
    if len(tfmd_gdf) < 1000:
        tfmd_gdf.to_file(outpath/f'{tile_fn}.geojson')
    else:
        tfmd_gdf.to_file(outpath/f'{tile_fn}.gpkg')