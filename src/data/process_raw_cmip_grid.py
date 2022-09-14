""" 
Data processing script: Create gridmaps for each model
Objective: create a mapping between the target grid from the yield dataset and the resolutions in the climate models
Input: 1 netcdf4 file with variable for each model 
Output: 1 csv with gridmap (lonxlat) between 0.5 (yield) resolution and model resolution
"""

import pandas as pd
import xarray as xr
import sys
import numpy as np
import os
cwd = os.getcwd()
sys.path.insert(0, cwd)
import h_geo as mp
import src.h_geo as mp

datapath = cwd + "/data/"
raw_path = datapath + "raw/"
clim_path = raw_path + "clim_mod/"
int_path = datapath + "interrim/"
processed_path = datapath + "processed/"

models = ['MIROC6', 'ACCESS-ESM1-5', 'CanESM5'] #, 'EC-Earth3-Veg']

version = {'MIROC6': 'v20191016',
        'EC-Earth3-Veg' : 'v20200225',
        'ACCESS-ESM1-5':  'v20191115',
        'CanESM5':  'v20190429'}

# Historical
for model in models:
    print(model)

    yieldmask = pd.read_csv(int_path + 'yieldmask.csv', index_col=0)

    model_path = clim_path + model + '/year/'
    path = model_path + 'cddETCCDI_yr_' + model + '_historical_r1i1p1f1_no-base_' + version[model] + '_1850-2014_v2-0.nc'
    ds_out_mon = xr.open_dataset(path)
    ds_out_mon = ds_out_mon.sel(time=slice('1982-01-01', '1982-12-12'))

    ds_out_mon['lon'] = np.where(ds_out_mon.lon > 180, ds_out_mon.lon - 360, ds_out_mon.lon)
    df_model = ds_out_mon.to_dataframe().reset_index()
    def convert_ds(x, y): 
        return mp.convert(df_model['lon'], df_model['lat'], x, y)

    def convert_ds_wrapper(x): 
        return convert_ds(x['lon'], x['lat'])

    print('start conversion of: ' + model)
    yieldmask['tuple'] = yieldmask.apply(convert_ds_wrapper, axis=1)
    yieldmask['lon2'] = yieldmask['tuple'].apply(lambda t: t[0])
    yieldmask['lat2'] = yieldmask['tuple'].apply(lambda t: t[1])
    yieldmask.to_csv(int_path + 'cmip/' + 'gridmap_' + model +'.csv')

# Simulations
for model in models:
    print(model)
    yieldmask = pd.read_csv(int_path + 'yieldmask.csv', index_col=0)
    model_path = clim_path + model + '/real/'
    path = model_path + "tas_Amon_"+model+"_ssp126_r1i1p1f1_gn_201501-210012.nc"
    ds_out_mon = xr.open_dataset(path)
    ds_out_mon = ds_out_mon.sel(time=slice('2020-01-01', '2020-12-12'))
    ds_out_mon['lon'] = np.where(ds_out_mon.lon > 180, ds_out_mon.lon - 360, ds_out_mon.lon)
    df_model = ds_out_mon.to_dataframe().reset_index()
    def convert_ds(x, y): 
        return mp.convert(df_model['lon'], df_model['lat'], x, y)
    def convert_ds_wrapper(x): 
        return convert_ds(x['lon'], x['lat'])
    print('start conversion of: ' + model)
    yieldmask['tuple'] = yieldmask.apply(convert_ds_wrapper, axis=1)
    yieldmask['lon2'] = yieldmask['tuple'].apply(lambda t: t[0])
    yieldmask['lat2'] = yieldmask['tuple'].apply(lambda t: t[1])
    yieldmask.to_csv(int_path + 'cmip/' + 'gridmap_sim_' + model +'.csv')


