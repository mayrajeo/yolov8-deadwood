import geopandas as gpd 
import shapely
import numpy as np

"""GREEDYNMM from sahi, implemented to work with geodataframes
and shapely polygons
"""

def poly_IoU(poly_1:shapely.Geometry, poly_2:shapely.Geometry) -> float:
    "Return Intersection-over-Union of two shapely.Geometry objects"
    inter = poly_1.intersection(poly_2)
    union = poly_1.union(poly_2)
    return inter.area / union.area

def poly_IoS(poly_1:shapely.Geometry, poly_2:shapely.Geometry) -> float:
    "Return Intersection-over-Minimum of two shapely.Geometry objects"
    inter = poly_1.intersection(poly_2).area
    min_area = min(poly_1.area, poly_2.area)
    return inter/min_area

def batched_greedy_nmm(
    preds:gpd.GeoDataFrame,
    match_metric:str='IOU',
    match_threshold:float=0.2
) -> gpd.GeoDataFrame:
    "Process GeoDataFrame containing predictions. Should have columns `label`, `score` and `geometry`"
    category_ids = preds.label.unique()

    out_gdf = gpd.GeoDataFrame(columns=preds.columns)

    for cat_id in category_ids:

        # Instead of only indices, get GeoDataFrame
        curr_cats = preds[preds.label == cat_id].copy()
        curr_keep_to_merge_list = greedy_nmm(curr_cats, match_metric, match_threshold)
        
        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_cats.iloc[curr_keep].copy()
            merge_list = [curr_cats.iloc[curr_merge_ind] for curr_merge_ind in curr_merge_list]

            if len(merge_list) > 0:
                geoms_to_merge = [c.geometry.buffer(0) for c in merge_list]
                keep.geometry = shapely.ops.unary_union([keep.geometry.buffer(0)] + geoms_to_merge)
            out_gdf.loc[len(out_gdf)] = keep
    out_gdf['score'] = out_gdf.score.astype(float)
    out_gdf['label'] = out_gdf.label.astype(int)
    out_gdf.crs = preds.crs
    return out_gdf

def greedy_nmm(    
    preds:gpd.GeoDataFrame,
    match_metric:str='IOU',
    match_threshold:float=0.2
) -> dict:
    
    keep_to_merge_list = {}
    scores = preds.score.values
    order = scores.argsort()
    
    geoms = [g.buffer(0) for g in preds.geometry.values]

    while len(order) > 0:
        idx = order[-1]
        order = order[:-1]

        if len(order) == 0:
            keep_to_merge_list[idx.tolist()] = []
            break
        
        intersecting_idx = np.array([i for i in order if geoms[i].intersects(geoms[idx])])
        if len(intersecting_idx) == 0:
            keep_to_merge_list[idx.tolist()] = []
            continue

        if match_metric == 'IOU':
            match_metrics = np.array([poly_IoU(geoms[idx], geoms[k]) 
                                      if geoms[k].intersects(geoms[idx]) else 0
                                      for k in intersecting_idx])
        elif match_metric == 'IOS':
            match_metrics = np.array([poly_IoS(geoms[idx], geoms[k])
                                      if geoms[k].intersects(geoms[idx]) else 0
                                      for k in intersecting_idx])
        else: 
            raise ValueError()
        
        mask = match_metrics <= match_threshold
        matched = np.flip(intersecting_idx[(mask==False).nonzero()[0]])
        unmatched = np.array([o for o in order if o not in matched])

        if len(unmatched) > 0:
            order = unmatched[scores[unmatched].argsort()]
        else:
            order = []

        keep_to_merge_list[idx.tolist()] = []

        for matched_ind in matched:
            keep_to_merge_list[idx.tolist()].append(matched_ind)
    return keep_to_merge_list
