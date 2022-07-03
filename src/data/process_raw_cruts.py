import os.path

if __name__ == '__main__': print('running process_raw_cruts.py')

import sys

sys.path.append("..")  # sys.path.append("..") # Adds higher directory to python modules path.

import pandas as pd
import xarray as xr
from os.path import exists

# TODO: Set relative path if needed (doesnt work right now but maybe for server)
# datapath = sys.path[len(sys.path) - 1] + "/data/"
datapath = 'C:\\Users\\langh\\OneDrive\\Documents\\Imperial\\Individual_project\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

#### CRUTS ####
print("Processing CRUTS")

vars = ['cld', 'dtr', 'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']
coords = ['lon', 'lat', 'month', 'year']

path_cruts = rawpath + 'CRUTS/'

path_var = path_cruts + 'cru_ts4.06.1901.2021.tmp.dat.nc/cru_ts4.06.1901.2021.tmp.dat.nc'
path_var_save = interrimpath + 'cruts_var/' + 'cruts_' + 'tmp' + '.csv'

if exists(path_var_save):
    df_cruts_out = pd.read_csv(path_var_save, index_col=0)
else:
    df_cruts_out = xr.open_dataset(path_var).to_dataframe().reset_index()
    df_cruts_out = df_cruts_out[df_cruts_out["time"] > '1980-01-16']
    df_cruts_out = df_cruts_out[df_cruts_out["time"] < '2017-01-16']
    df_cruts_out['month'] = pd.DatetimeIndex(df_cruts_out['time']).month
    df_cruts_out['year'] = pd.DatetimeIndex(df_cruts_out['time']).year
    mainout = coords + ['tmp']
    df_cruts_out = df_cruts_out[mainout]
    df_cruts_out.to_csv(path_var_save)

df_cruts_out.set_index(['lat', 'lon', 'month', 'year'], inplace=True)

for var in vars:
    print(var)
    path_var = path_cruts + 'cru_ts4.06.1901.2021.' + var + '.dat.nc\\cru_ts4.06.1901.2021.' + var + '.dat.nc'
    path_var_save = interrimpath + 'cruts_var/' + 'cruts_' + var + '.csv'
    if exists(path_var_save):
        df_var = pd.read_csv(path_var_save, index_col=0)
    else:
        df_var = xr.open_dataset(path_var).to_dataframe().reset_index()
        df_var = df_var[df_var["time"] > '1980-01-16']
        df_var = df_var[df_var["time"] < '2017-01-16']
        df_var['month'] = pd.DatetimeIndex(df_var['time']).month
        df_var['year'] = pd.DatetimeIndex(df_var['time']).year
        var_out = coords + [var]
        df_var = df_var[var_out]
        df_var.to_csv(path_var_save)
    df_var.set_index(['lat', 'lon', 'month', 'year'], inplace=True)
    df_cruts_out = df_cruts_out.join(df_var, how='outer', on=coords)

df_cruts_out['wet'] = df_cruts_out['wet'].dt.days
df_cruts_out['frs'] = df_cruts_out['frs'].dt.days

df_cruts_out.to_csv(interrimpath + 'df_cruts_mth.csv')

print('Done!')

del df_cruts_out, df_var
