""" 
Data processing script: Process raw etccdi files
Objective: Combine and standardise the raw etccdi files into collected dataframes
Input: directories with netcdf files for each model-scenario-variable in structure: model/time/file.netcdf where time is month or year
Output: csv files with dataframes for each model-scenario and average across models
"""

from genericpath import exists
import xarray as xr
import pandas as pd
import numpy as np
import sys
import os
cwd = os.getcwd()
sys.path.insert(0, cwd)

from src.h_utils import *
from src.h_utils import df_mth_to_year

cwd = os.getcwd()
sys.path.insert(0, cwd)

datapath = cwd + "/data/"
raw_path = datapath + "raw/"
clim_path = raw_path + "clim_mod/"
int_path = datapath + "interrim/"
processed_path = datapath + "processed/"

clim_path = raw_path + "clim_mod/"

month_vars = ['rx1day', 'rx5day', 'tn10p', 'tn90p',
              'tnn', 'tnx', 'tx10p', 'tx90p', 'txn', 'txx']

year_vars = ['csdi', 'cwd', 'fd', 'gsl', 'id', 'prcptot',
             'r1mm', 'r10mm', 'r20mm', 'r95p', 'r99p', 'sdii',
             'su', 'tr', 'wsdi']


models = ['MIROC6', 'ACCESS-ESM1-5', 'CanESM5'] #, 'EC-Earth3-Veg']

version = {'MIROC6': 'v20191016',
        'EC-Earth3-Veg' : 'v20200225',
        'ACCESS-ESM1-5':  'v20191115',
        'CanESM5':  'v20190429'}

scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585'] # 'hist', 

def base(var):
    base_vars = ['dtr', 'rx1day', 'rx5day', 'tnn', 'tnx', 'txn', 'txx', 'cdd', 'cwd', 'fd', 'gsl', 'id', 'prcptot', 'r1mm', 'r10mm', 'r20mm', 'sdii', 'su', 'tr']
    if var in base_vars:
        return 'no-base'
    else:
        return 'b1961-1990'

def days_to_int(df):
    for col in df.columns:
        if df[col].dtype == '<m8[ns]':
                df[col] = df[col].dt.days
    return df

def ds_to_df(ds, gridmap, m = False):
    ds_out = ds.drop(['lat_bnds', 'lon_bnds', 'time_bnds', 'height'])
    ds_out['lon'] = np.where(ds_out.lon > 180, ds_out.lon - 360, ds_out.lon)
    df_out = ds_out.to_dataframe().reset_index()
    df_out.rename(columns = {'lon':'lon2', 'lat':'lat2'}, inplace = True)
    df_out = df_out.merge(gridmap, on=['lon2', 'lat2'])
    df_out.drop(['tuple', 'lon2', 'lat2'], axis=1, inplace=True)
    if m:
        df_out['month'] = df_out.apply(lambda x : x.time.month, axis=1)
    df_out['year'] = df_out.apply(lambda x : x.time.year, axis=1)
    df_out.drop('time', axis=1, inplace=True)
    df_out = df_out.reset_index(drop=True)
    return df_out

for scenario in scenarios:
    for model in models:
        # ETCCDI is already pre-processed by copernicus so 
        # jump straight to final repo to save disk space
        outpath = processed_path + model +'_'+ scenario +'_pred_etccdi.csv'

        if exists(outpath):
            print('Skipping', model, scenario)
        else:
            gridmap = pd.read_csv(int_path + 'cmip/' + 'gridmap_'+ model + '.csv', index_col=0)
            model_path = clim_path + model +'/'
            print(model, scenario)
            
            # Month
            var = 'dtr'
            if scenario == 'historical':
                path = model_path + 'month/' + var + "ETCCDI_mon_"+ model +"_historical_r1i1p1f1_" + \
                    base(var) + "_" + version[model] +"_185001-201412_v2-0.nc"
            elif model == 'CanESM5' and scenario == 'ssp126':
                path = model_path + 'month/' + var + "ETCCDI_mon_"+ model +"_" + scenario + "_r1i1p1f1_" + \
                        base(var) + '_' + version[model] +'_201501-230012_v2-0.nc'
            else:
                path = model_path + 'month/' + var + "ETCCDI_mon_"+ model +"_" + scenario + "_r1i1p1f1_" + \
                    base(var) + '_' + version[model] +'_201501-210012_v2-0.nc'
            
            ds_out_mon = xr.open_dataset(path)
            if scenario == 'historical':
                ds_out_mon = ds_out_mon.sel(time=slice('1980-01-16', '2014-12-16'))
            else:
                ds_out_mon = ds_out_mon.sel(time=slice('2014-01-16', '2100-12-30'))

            for var in month_vars:
                print(var)
                if scenario == 'historical':
                    path = model_path + 'month/' + var + "ETCCDI_mon_"+ model +"_historical_r1i1p1f1_" + \
                        base(var) + "_" + version[model] +"_185001-201412_v2-0.nc"
                elif model == 'CanESM5' and scenario == 'ssp126':
                    path = model_path + 'month/' + var + "ETCCDI_mon_"+ model +"_" + scenario + "_r1i1p1f1_" + \
                        base(var) + '_' + version[model] +'_201501-230012_v2-0.nc'
                else:
                    path = model_path + 'month/' + var + "ETCCDI_mon_"+ model +"_" + scenario + "_r1i1p1f1_" + \
                        base(var) + '_' + version[model] +'_201501-210012_v2-0.nc'
                ds_mon = xr.open_dataset(path)
                if scenario == 'historical':
                    ds_mon = ds_mon.sel(time=slice('1980-01-16', '2014-12-16'))
                else:
                    ds_mon = ds_mon.sel(time=slice('2014-01-16', '2100-12-30'))

                ds_out_mon = ds_out_mon.merge(ds_mon)
            
            df_out_mon = ds_to_df(ds_out_mon, gridmap, True)
            
            # Year
            var = 'cdd'
            if scenario == 'historical':
                path = model_path + 'year/' + var + "ETCCDI_yr_"+ model +"_historical_r1i1p1f1_" + \
                    base(var) + "_" + version[model] +"_1850-2014_v2-0.nc"
            elif model == 'CanESM5' and scenario == 'ssp126':
                    path = model_path + 'year/' + var + "ETCCDI_yr_"+ model +"_" + scenario + "_r1i1p1f1_" + \
                    base(var) + '_' + version[model] +'_2015-2300_v2-0.nc'
            else:
                path = model_path + 'year/' + var + "ETCCDI_yr_"+ model +"_" + scenario +"_r1i1p1f1_" + \
                    base(var) + '_' + version[model] +'_2015-2100_v2-0.nc'

            ds_out_yr = xr.open_dataset(path)
            if scenario == 'historical':
                ds_out_yr = ds_out_yr.sel(time=slice('1980-01-16', '2014-12-16'))
            else:
                ds_out_yr = ds_out_yr.sel(time=slice('1980-01-16', '2100-12-30'))
                
            for var in year_vars:
                print(var)
                if scenario == 'historical':
                    path = model_path + 'year/' + var + "ETCCDI_yr_"+ model +"_historical_r1i1p1f1_" + \
                        base(var) + "_" + version[model] +"_1850-2014_v2-0.nc"
                elif model == 'CanESM5' and scenario == 'ssp126':
                    path = model_path + 'year/' + var + "ETCCDI_yr_"+ model +"_" + scenario + "_r1i1p1f1_" + \
                        base(var) + '_' + version[model] +'_2015-2300_v2-0.nc'
                else:
                    path = model_path + 'year/' + var + "ETCCDI_yr_"+ model +"_" + scenario +"_r1i1p1f1_" + \
                        base(var) + '_' + version[model] +'_2015-2100_v2-0.nc'

                ds_yr = xr.open_dataset(path)
                if scenario == 'historical':
                    ds_yr = ds_yr.sel(time=slice('1980-01-16', '2014-12-16'))
                else:
                    ds_yr = ds_yr.sel(time=slice('1980-01-16', '2100-12-30'))
                ds_out_yr = ds_out_yr.merge(ds_yr)

            df_out_yr = ds_to_df(ds_out_yr, gridmap)
            df_out_mon.columns = df_out_mon.columns.str.rstrip("ETCCDI")
            df_out_yr.columns = df_out_yr.columns.str.rstrip("ETCCDI")
            df_out_mthyr = df_mth_to_year(df_out_mon)
            df_all = df_out_mthyr.merge(df_out_yr, on=['lon', 'lat', 'year'])
            df_all = days_to_int(df_all)
            df_all.to_csv(outpath)

# Combine all the models to an average output:
print('Making averages')
for scenario in scenarios:
    outpath = processed_path + 'avg_'+ scenario +'_pred_etccdi.csv'

    if exists(outpath):
        print('done already - avg ', scenario)
    else:
        print('computing avg', scenario)
        df_ac = pd.read_csv(processed_path + 'ACCESS-ESM1-5' + '_' + scenario + '_pred_etccdi.csv', index_col=0)
        df_ca = pd.read_csv(processed_path + 'CanESM5' + '_' + scenario + '_pred_etccdi.csv', index_col=0)
        df_mi = pd.read_csv(processed_path + 'MIROC6' + '_' + scenario + '_pred_etccdi.csv', index_col=0)

        collected = gc.collect()
        print("Garbage collector: collected {} objects.".format(collected))
        
        df_ac.sort_values(['lon', 'lat', 'year'], inplace=True)
        df_ca.sort_values(['lon', 'lat', 'year'], inplace=True)
        df_mi.sort_values(['lon', 'lat', 'year'], inplace=True)
        
        df_ac.set_index(['lon', 'lat', 'year'], inplace = True)
        df_ca.set_index(['lon', 'lat', 'year'], inplace = True)
        df_mi.set_index(['lon', 'lat', 'year'], inplace = True)
        
        df_concat = pd.concat((df_ac, df_ca, df_mi))
        df_out = df_concat.groupby(df_concat.index).mean()
        df_out = df_out.reset_index()
        
        df_out['lon'] = df_out['index'].apply(lambda x: x[0])
        df_out['lat'] = df_out['index'].apply(lambda x: x[1])
        df_out['year'] = df_out['index'].apply(lambda x: x[2])
        df_out.drop('index', inplace = True, axis = 1)
    
        df_out.to_csv(outpath)





