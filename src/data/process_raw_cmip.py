""" 
Data processing script: Process cmip6 files
Objective: Combine and standardise the raw cmip6 files into collected dataframes
Input: directories with netcdf files for each model-scenario-variable in structure: model/file.netcdf
Output: csv files with dataframes for each model-scenario
"""

from genericpath import exists
import xarray as xr
import pandas as pd
import numpy as np
import sys
import os
import gc
cwd = os.getcwd()
sys.path.insert(0, cwd)

current_directory = os.getcwd()
datapath = current_directory + "/data/"
raw_path = datapath + "raw/"
clim_path = raw_path + "clim_mod/"
int_path = datapath + "interrim/"

models = ['MIROC6', 'ACCESS-ESM1-5', 'CanESM5']  # , 'EC-Earth3-Veg']

scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585'] 

vars = ['tasmin', 'tasmax', 'pr', 'clt', 'prw']

miroc_years = ['20250101-20341231', '20350101-20441231', '20450101-20541231', '20550101-20641231',
               '20650101-20741231', '20750101-20841231', '20850101-20941231', '20950101-21001231']

miroc_years_hist = ['19900101-19991231', '20000101-20091231', '20100101-20141231']


model_hist = {'MIROC6':'195001-201412', 'CanESM5': '185001-201412', 'ACCESS-ESM1-5': '185001-201412'}

def _min_max(x):
    x = x - 30
    return np.where(x > 0, x, 0)

def ds_to_df_day(ds_day, gridmap):
    ds_day['lon'] = np.where(ds_day.lon > 180, ds_day.lon - 360, ds_day.lon)
    ds_day = ds_day.drop_vars(
        ['lon_bnds', 'lat_bnds', 'height', 'time_bnds'], errors='raise')
    ds_day['tasmax'] = ds_day['tasmax'] - 272.15
    ds_day = ds_day.apply(_min_max)
    year_month_idx = pd.MultiIndex.from_arrays(
        [ds_day['time.year'].to_series(), ds_day['time.month'].to_series()])
    ds_day.coords['year_month'] = ('time', year_month_idx)
    ds_day = ds_day.groupby('year_month').sum()
    df_out = ds_day.to_dataframe().reset_index()
    df_out.rename(columns={'year_month_level_0': 'year',
                  'year_month_level_1': 'month', 'tasmax': 'heat_mth'}, inplace = True)
    df_out.rename(columns={'lon': 'lon2', 'lat': 'lat2'}, inplace=True)
    df_out = df_out.merge(gridmap, on=['lon2', 'lat2'])
    df_out.drop(['tuple', 'lon2', 'lat2'], axis=1, inplace=True)
    df_out = df_out.sort_values(by=['lon', 'lat', 'year', 'month'])
    df_out['heat6mth'] = df_out['heat_mth'].rolling(6).sum()
    return df_out

def ds_to_df_mon(ds, gridmap):
    ds_out = ds.drop(['lat_bnds', 'lon_bnds', 'time_bnds', 'height'])
    
    ds_out['lon'] = np.where(ds_out.lon > 180, ds_out.lon - 360, ds_out.lon)

    ds_out.coords['year'] = ds_out['time.year'].to_series()
    ds_out.coords['month'] =ds_out['time.month'].to_series()
    ds_out = ds_out.drop_vars('time')
    df_out = ds_out.to_dataframe().reset_index()
    df_out.rename(columns={'lon': 'lon2', 'lat': 'lat2'}, inplace=True)
    df_out = df_out.merge(gridmap, on=['lon2', 'lat2'])
    df_out.drop(['tuple', 'lon2', 'lat2'], axis=1, inplace=True)
    df_out = df_out.reset_index(drop=True)
    return df_out

for scenario in scenarios:
    for model in models:
        print(model, scenario)  
        model_path = clim_path + model + '/cruts/'
        outpath = int_path + 'cmip/' + model + '_' + scenario + '_cruts.csv'
        gridmap = pd.read_csv(int_path + 'cmip/' +
                              'gridmap_sim_' + model + '.csv', index_col=0)
        if exists(outpath):
            print('Skipping', model, scenario)
        else:
            var = 'tas'
            if scenario == 'historical':
                path = model_path + var + '_Amon_'+ model +'_historical_r1i1p1f1_gn_'+ model_hist[model] + '.nc'
            else:
                path = model_path + var + "_Amon_"+model+"_" + \
                    scenario + "_r1i1p1f1_gn_201501-210012.nc"
            ds_out_mon = xr.open_dataset(path)
            for var in vars:
                if scenario == 'historical':
                    path = model_path + var + '_Amon_'+ model +'_historical_r1i1p1f1_gn_'+ model_hist[model] + '.nc'
                else:
                    path = model_path + var + "_Amon_"+model+"_" + \
                        scenario + "_r1i1p1f1_gn_201501-210012.nc"
                print(var)
                ds_mon = xr.open_dataset(path)
                ds_out_mon = ds_out_mon.merge(ds_mon)
            ds_out_mon = ds_out_mon.sel(time=slice('1982-01-01', '2100-12-30'))
            df_out_mon = ds_to_df_mon(ds_out_mon, gridmap)
            print('day var')
            if scenario == 'historical':
                if model == 'ACCESS-ESM1-5':
                    path1 = model_path + 'tasmax_day_ACCESS-ESM1-5_' + \
                        scenario+'_r1i1p1f1_gn_19500101-19991231.nc'
                    path2 = model_path + 'tasmax_day_ACCESS-ESM1-5_' + \
                        scenario+'_r1i1p1f1_gn_20000101-20141231.nc'
                    ds_day1 = xr.open_dataset(path1)
                    ds_day1 = ds_day1.sel(time=slice('1982-01-01', '2100-12-30'))
                    ds_day2 = xr.open_dataset(path2)
                    ds_day = ds_day2.merge(ds_day1)
                elif model == 'MIROC6':
                    path = model_path + 'tasmax_day_MIROC6_' + \
                        scenario+'_r1i1p1f1_gn_19800101-19891231.nc'
                    ds_day = xr.open_dataset(path)
                    for year in miroc_years_hist:
                        path1 = model_path + 'tasmax_day_MIROC6_' + \
                            scenario+'_r1i1p1f1_gn_' + year + '.nc'
                        ds_day1 = xr.open_dataset(path1)
                        ds_day = ds_day.merge(ds_day1)
                        ds_day = ds_day.sel(time=slice('1982-01-01', '2100-12-30'))
                else:
                    path = model_path +'tasmax_day_CanESM5_'+scenario + '_r1i1p1f1_gn_18500101-20141231.nc'
                    ds_day = xr.open_dataset(path)
                    ds_day = ds_day.sel(time=slice('1982-01-01', '2100-12-30'))
            else:
                if model == 'ACCESS-ESM1-5':
                    path1 = model_path + 'tasmax_day_ACCESS-ESM1-5_' + \
                        scenario+'_r1i1p1f1_gn_20150101-20641231.nc'
                    path2 = model_path + 'tasmax_day_ACCESS-ESM1-5_' + \
                        scenario+'_r1i1p1f1_gn_20650101-21001231.nc'
                    ds_day1 = xr.open_dataset(path1)
                    ds_day2 = xr.open_dataset(path2)
                    ds_day = ds_day2.merge(ds_day1)
                elif model == 'MIROC6':
                    path = model_path + 'tasmax_day_MIROC6_' + \
                        scenario+'_r1i1p1f1_gn_20150101-20241231.nc'
                    ds_day = xr.open_dataset(path)
                    for year in miroc_years:
                        path1 = model_path + 'tasmax_day_MIROC6_' + \
                            scenario+'_r1i1p1f1_gn_' + year + '.nc'
                        ds_day1 = xr.open_dataset(path1)
                        ds_day = ds_day.merge(ds_day1)
                else:
                    path = model_path +'tasmax_day_CanESM5_'+scenario + '_r1i1p1f1_gn_20150101-21001231.nc'
                    ds_day = xr.open_dataset(path)

            df_day = ds_to_df_day(ds_day, gridmap)
            df_out = df_out_mon.merge(
                df_day, on=['lon', 'lat', 'year', 'month'])
            try:
                df_out.drop('time', inplace = True)
            except:
                pass

            df_out.to_csv(outpath)
            collected = gc.collect()
            print("Garbage collector: collected {} objects.".format(collected))
        
