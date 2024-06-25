from geo2ml.data.tiling import Tiler
from geo2ml.data.cv import *
from pathlib import Path 
import os

from fastcore.script import *

@call_parse
def make_data(datapath:Path, # Base datapath where everything else is
              ):
    "Generate dataset from the raw data"

    cats = ['groundwood', 'uprightwood']
    # Set paths
    hp_inpath = datapath/'raw/hiidenportti/virtual_plots'
    spk_inpath = datapath/'raw/sudenpesankangas/virtual_plots'
    outpath = datapath/'yolov8'

    # Make Sudenpesänkangas data

    # List files to chip
    hp_files_to_chip = [hp_inpath/f'{t}/images/{f}' for t in ['train', 'valid', 'test'] 
                        for f in os.listdir(hp_inpath/f'{t}/images')]
    hp_target_column = 'layer'
    # Chip and convert the data
    for f in hp_files_to_chip:
        tilepath = outpath/f.stem
        os.makedirs(tilepath, exist_ok=True)
        vectorpath = str(f).replace('images', 'vectors').replace('tif', 'geojson')
        tiler = Tiler(outpath=tilepath, gridsize_x=640, gridsize_y=640, overlap=(0,0))
        tiler.tile_raster(f, allow_partial_data=True)
        tiler.tile_vector(vectorpath, min_area_pct=0)
        shp_to_yolo(tilepath/'images', 
                    tilepath/'vectors', 
                    tilepath, 
                    label_col=hp_target_column,
                    names=cats, 
                    ann_format='polygon', 
                    min_bbox_area=0)
        
    # Create text files for train, val and test sets
        
    hp_train_tiles = [t[:-4] for t in os.listdir(hp_inpath/'train/images')]
    hp_val_tiles = [t[:-4] for t in os.listdir(hp_inpath/'valid/images')]
    hp_test_tiles = [t[:-4] for t in os.listdir(hp_inpath/'test/images')]

    hp_train_files = []
    for t in hp_train_tiles:
        tile_files = os.listdir(outpath/t/'images')
        hp_train_files.extend([os.path.abspath(outpath/t/'images'/f) for f in tile_files])

    with open(outpath/'hp_train.txt', 'w') as f:
        for t in hp_train_files: f.write(t+'\n')

    hp_val_files = []
    for t in hp_val_tiles:
        tile_files = os.listdir(outpath/t/'images')
        hp_val_files.extend([os.path.abspath(outpath/t/'images'/f) for f in tile_files])

    with open(outpath/'hp_val.txt', 'w') as f:
        for t in hp_val_files: f.write(t+'\n')

    hp_test_files = []
    for t in hp_test_tiles:
        tile_files = os.listdir(outpath/t/'images')
        hp_test_files.extend([os.path.abspath(outpath/t/'images'/f) for f in tile_files])

    with open(outpath/'hp_test.txt', 'w') as f:
        for t in hp_test_files: f.write(t+'\n')

    # Make Sudenpesänkangas data

    # List files to chip
    spk_files_to_chip = [spk_inpath/f'{t}/images/{f}' for t in ['train', 'valid', 'test'] 
                         for f in os.listdir(spk_inpath/f'{t}/images')]
    spk_target_column = 'label'
    # Chip and convert the data
    for f in spk_files_to_chip:
        tilepath = outpath/f.stem
        os.makedirs(tilepath, exist_ok=True)
        vectorpath = str(f).replace('images', 'vectors').replace('tif', 'geojson')
        tiler = Tiler(outpath=tilepath, gridsize_x=640, gridsize_y=640, overlap=(0,0))
        tiler.tile_raster(f, allow_partial_data=True)
        tiler.tile_vector(vectorpath, min_area_pct=0)
        shp_to_yolo(tilepath/'images', 
                    tilepath/'vectors', 
                    tilepath, 
                    label_col=spk_target_column,
                    names=cats, 
                    ann_format='polygon', 
                    min_bbox_area=0)
        
    # Create text files for train, val and test sets
        
    spk_train_tiles = [t[:-4] for t in os.listdir(spk_inpath/'train/images')]
    spk_val_tiles = [t[:-4] for t in os.listdir(spk_inpath/'valid/images')]
    spk_test_tiles = [t[:-4] for t in os.listdir(spk_inpath/'test/images')]

    spk_train_files = []
    for t in spk_train_tiles:
        tile_files = os.listdir(outpath/t/'images')
        spk_train_files.extend([os.path.abspath(outpath/t/'images'/f) for f in tile_files])

    with open(outpath/'spk_train.txt', 'w') as f:
        for t in spk_train_files: f.write(t+'\n')

    spk_val_files = []
    for t in spk_val_tiles:
        tile_files = os.listdir(outpath/t/'images')
        spk_val_files.extend([os.path.abspath(outpath/t/'images'/f) for f in tile_files])

    with open(outpath/'spk_val.txt', 'w') as f:
        for t in spk_val_files: f.write(t+'\n')

    spk_test_files = []
    for t in spk_test_tiles:
        tile_files = os.listdir(outpath/t/'images')
        spk_test_files.extend([os.path.abspath(outpath/t/'images'/f) for f in tile_files])

    with open(outpath/'spk_test.txt', 'w') as f:
        for t in spk_test_files: f.write(t+'\n')
