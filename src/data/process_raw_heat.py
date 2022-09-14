""" 
Data processing script: Process the CPC files files
Objective: Combine and compute the heat values for each crop
Input: directories with netcdf files for each crop-year in structure: crop/yield_year.netcdf
Output: csv files with dataframes for each crop and 1 combined file, 1 yieldmask file with individual entries for all non-zero pixels
"""


if __name__ == '__main__':
    print('running process_cpc_raw.py')    

import xarray as xr
import pandas as pd
import sys
import os
cwd = os.getcwd()
sys.path.insert(0, cwd)
from h_utils import recenter_lon, max_zero

datapath = cwd + "/data/"
raw_path = datapath + "raw/"
clim_path = raw_path + "clim_mod/"
int_path = datapath + "interrim/"
processed_path = datapath + "processed/"

## CPC TMAX ###
print("Processing CPC")

path_cpc = raw_path + "cpc/tmax."

thresholds = 30 #OBS In paper, use the pre-calcualted individual crop thresholds

df_tmax = xr.open_dataset(path_cpc + str(1981) +
                          '.nc').to_dataframe().reset_index()

df_tmax['heat'] = (df_tmax['tmax'] - thresholds).transform(max_zero)

df_tmax['month'] = pd.DatetimeIndex(df_tmax['time']).month
df_tmax['year'] = pd.DatetimeIndex(df_tmax['time']).year

df_tmax_out = df_tmax.groupby(["lat", "lon", "month", "year"])[["heat"]] \
        .sum().reset_index()

df_tmax_out['lon'] = df_tmax_out['lon'].apply(recenter_lon)
yieldmask = pd.read_csv(int_path + 'yieldmask.csv', index_col=0)
df_tmax_out = df_tmax_out.merge(yieldmask, on = ['lon', 'lat'], how = 'inner')

for year in range(1982, 2017):
    print(year)
    df_year = xr.open_dataset(
        path_cpc + str(year) + '.nc').to_dataframe().reset_index()
    df_year['month'] = pd.DatetimeIndex(df_year['time']).month
    df_year['year'] = pd.DatetimeIndex(df_year['time']).year

    df_year['heat'] = (df_year['tmax'] - thresholds).transform(max_zero)

    df_year = df_year.groupby(["lat", "lon", "month", "year"])[["heat"]] \
        .sum().reset_index()

    df_year['lon'] = df_year['lon'].apply(recenter_lon)
    df_year = df_year.merge(yieldmask, on = ['lon', 'lat'], how = 'inner')
    df_tmax_out = pd.concat([df_tmax_out, df_year], sort=False)
    df_tmax_out.to_csv(int_path + 'cpc_interrim.csv')

df_tmax_out = df_tmax_out.sort_values(by=['lon', 'lat', 'year', 'month'])
df_tmax_out['heat6mth'] = df_tmax_out['heat_mth'].rolling(6).sum()
df_tmax_out['lon'] = df_tmax_out['lon'].apply(recenter_lon)
df_tmax_out.to_csv(int_path + 'df_heat_mth.csv')
