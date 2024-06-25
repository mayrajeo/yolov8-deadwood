from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import os

from fastcore.script import *

def add_results_to_df(metrics, df, model):
    df.loc[len(df)] = {
        'model': model,
        'class_name': 'groundwood',
        'precision(B)': metrics.box.p[0],
        'recall(B)': metrics.box.r[0],
        'mAP50(B)': metrics.box.ap50[0],
        'mAP50-95(B)': metrics.box.ap[0],
        'precision(M)': metrics.seg.p[0],
        'recall(M)': metrics.seg.r[0],
        'mAP50(M)': metrics.seg.ap50[0],
        'mAP50-95(M)': metrics.seg.ap[0]
    }

    df.loc[len(df)] = {
        'model': model,
        'class_name': 'uprightwood',
        'precision(B)': metrics.box.p[1],
        'recall(B)': metrics.box.r[1],
        'mAP50(B)': metrics.box.ap50[1],
        'mAP50-95(B)': metrics.box.ap[1],
        'precision(M)': metrics.seg.p[1],
        'recall(M)': metrics.seg.r[1],
        'mAP50(M)': metrics.seg.ap50[1],
        'mAP50-95(M)': metrics.seg.ap[1]
    }

    df.loc[len(df)] = {
        'model': model,
        'class_name': 'all',
        'precision(B)': metrics.mean_results()[0],
        'recall(B)': metrics.mean_results()[1],
        'mAP50(B)': metrics.mean_results()[2],
        'mAP50-95(B)': metrics.mean_results()[3],
        'precision(M)':metrics.mean_results()[4],
        'recall(M)': metrics.mean_results()[5],
        'mAP50(M)': metrics.mean_results()[6],
        'mAP50-95(M)': metrics.mean_results()[7],
    }
    return

@call_parse
def make_result_report(datapath:Path, # Path where the yolo datasets are
                       rundir:Path, # Where are the run directorys for the yolov8 models
                       outpath:Path, # Where to save the results
                       ):
    # Set dataset paths
    hp_dataset = datapath/'hiidenportti_dw.yaml'
    spk_dataset = datapath/'spk_dw.yaml'
    both_dataset = datapath/'both.yaml'

    if not os.path.exists(outpath): os.makedirs(outpath)
    models = os.listdir(rundir)

    # Make dataframes for results
    res_columns=['model', 'class_name', 'precision(B)', 'recall(B)', 'mAP50(B)', 
                 'mAP50-95(B)', 'precision(M)', 'recall(M)', 'mAP50(M)', 'mAP50-95(M)']

    hp_val_results = pd.DataFrame(columns=res_columns)
    hp_test_results = pd.DataFrame(columns=res_columns)
    spk_val_results = pd.DataFrame(columns=res_columns)
    spk_test_results = pd.DataFrame(columns=res_columns)

    both_val_results = pd.DataFrame(columns=res_columns)
    both_test_results = pd.DataFrame(columns=res_columns)

    for m in models:
        model = YOLO(rundir/m/'weights/best.pt')
        # get hp results
        model.overrides['data'] = hp_dataset
        metrics = model.val(split='val')
        add_results_to_df(metrics, hp_val_results, m)
        metrics = model.val(split='test')
        add_results_to_df(metrics, hp_test_results, m)

        # get spk results
        model.overrides['data'] = spk_dataset
        metrics = model.val(split='val')
        add_results_to_df(metrics, spk_val_results, m)
        metrics = model.val(split='test')
        add_results_to_df(metrics, spk_test_results, m)

        # get combined results
        model.overrides['data'] = both_dataset
        metrics = model.val(split='val')
        add_results_to_df(metrics, both_val_results, m)
        metrics = model.val(split='test')
        add_results_to_df(metrics, both_test_results, m)
        del model

    hp_val_results.to_csv(outpath/'hp_val.csv', index=False)
    hp_test_results.to_csv(outpath/'hp_test.csv', index=False)
    spk_val_results.to_csv(outpath/'spk_val.csv', index=False)
    spk_test_results.to_csv(outpath/'spk_test.csv', index=False)
    both_val_results.to_csv(outpath/'both_val.csv', index=False)
    both_test_results.to_csv(outpath/'both_test.csv', index=False)
