from importlib.resources import path
import geopandas as gpd
from shapely.geometry import Point
from raster2xyz.raster2xyz import Raster2xyz
import pandas as pd
from src.utils import get_data, rmse, baseline_rolling, round_to_quarters
import numpy as np
import shapely

path_data = "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\"
path_raw = path_data + "raw\\harvest_area\\"
path_interrim = path_data + "interrim\\harvest\\"
path_tiff = path_raw + "tiff\\"

crop_s = {"wheat" : "whe", "maize" : "mze", "soy": "soy", "rice": "rcw"}
crops = ['wheat', 'soy', 'maize', 'rice']

for crop in crops:
    path_crop_in = path_tiff + crop_s[crop] + "_2010_har.tif"
    path_crop_out = path_raw + crop + "_2010_har.csv"
    rtxyz = Raster2xyz()
    rtxyz.translate(path_crop_in, path_crop_out)
    df_crop_har = pd.read_csv(path_crop_out)
    df_crop_har= df_crop_har.rename(columns={"x":"lon", "y": "lat", "z": "harvest"})
    df_crop_har['lon'] = df_crop_har['lon'].apply(round_to_quarters)
    df_crop_har['lat'] = df_crop_har['lat'].apply(round_to_quarters)
    df_crop_har_group = df_crop_har.groupby(['lon', 'lat']).sum()['harvest'].reset_index()
    df_crop_har_group = df_crop_har_group.loc[df_crop_har_group['harvest'] != 0]
    df_crop_har_group.to_csv(path_interrim + "df_" + crop +"_har.csv")


