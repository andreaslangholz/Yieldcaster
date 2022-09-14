""" 
Data processing script: Process raw cruts files
Objective: Combine and standardise the raw cruts files into 1 dataframe for modeltraining
Input: directory with netcdf files for each individual cruts feature
Output: csv file 1 dataframe containing all cruts features
"""

if __name__ == '__main__':
    print('running process_raw_cruts.py')

import pandas as pd
import xarray as xr
import gc
import os 
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd)

import src.h_utils as ut

current_directory = os.getcwd()
datapath = current_directory + "/data/"
raw_path = datapath + "raw/"
int_path = datapath + "interrim/"
path_cruts = raw_path + 'cruts/'

# Unzip files
zip = False

try:
    zip = sys.argv[1]
except:
    print('No unzipping')

z_vars = ['cld', 'tmp','tmn', 'tmx', 'vap', 'pre']

if zip:
    print("Unzipping")
    for var in z_vars:
        path_var = path_cruts + 'gz/' + 'cru_ts4.05.1901.2020.'+ var +'.dat.nc.gz'
        path_var_out = path_cruts + 'cruts_' + var +'.nc'
        ut.gunzip(path_var, path_var_out)

# Load yieldmask to subset on
coords = ['lon', 'lat', 'month', 'year']
yieldmask = pd.read_csv(int_path + 'yieldmask.csv', index_col=0)

# Do the first variable manually (could maybe be done smoother by concatenating..)
first_var = 'cld'

path_var = path_cruts + 'cruts_' + first_var +'.nc'
path_var_save = int_path + 'cruts_var/' + 'cruts_' + first_var + '.csv'

df_var = xr.open_dataset(path_var).to_dataframe().reset_index()
df_var = df_var[df_var["time"] > '1980-01-16']
df_var = df_var[df_var["time"] < '2017-01-16']
df_var = df_var.merge(yieldmask, on = ['lon', 'lat'], how = 'inner') 
df_var = df_var.dropna()
df_var['month'] = pd.DatetimeIndex(df_var['time']).month
df_var['year'] = pd.DatetimeIndex(df_var['time']).year
df_var.drop('time', axis=1, inplace = True)

var_out = coords + [first_var]
df_var = df_var[var_out]
df_var.to_csv(path_var_save)

print('first done!')

df_cruts_out = df_var
df_cruts_out.set_index(coords, inplace=True)

# Loop over remaining variables and join with dataframe
vars = ['tmp','tmn', 'tmx', 'vap', 'pre']

for var in vars:
    path_var = path_cruts + 'cruts_' + var +'.nc'
    path_var_save = int_path + 'cruts_var/' + 'cruts_' + var + '.csv'

    df_var = xr.open_dataset(path_var).to_dataframe().reset_index()
    df_var = df_var[df_var["time"] > '1980-01-16']
    df_var = df_var[df_var["time"] < '2017-01-16']
    df_var = df_var.merge(yieldmask, on = ['lon', 'lat'], how = 'inner') 
    df_var = df_var.dropna()
    df_var['month'] = pd.DatetimeIndex(df_var['time']).month
    df_var['year'] = pd.DatetimeIndex(df_var['time']).year
    df_var.drop('time', axis=1, inplace = True)

    var_out = coords + [var]
    df_var = df_var[var_out]
    df_var.to_csv(path_var_save)
    df_var.set_index(coords, inplace=True)
    df_cruts_out = df_cruts_out.join(df_var, how='outer')
    del df_var
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % (collected))

df_cruts_out = df_cruts_out.reset_index()
df_cruts_out.to_csv(int_path + 'df_cruts_mth.csv')

print('Done!')
del df_cruts_out
