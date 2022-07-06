import os.path

if __name__ == '__main__': print('running process_raw_cruts.py')

import sys

sys.path.append("..")  # sys.path.append("..") # Adds higher directory to python modules path.

import pandas as pd
import xarray as xr
from os.path import exists
# import src.utils as ut

try:
    import utils as ut
    print('utils loaded from ..')

except:
    print('no module in ..')
    try:
        import src.utils as ut
        print('utils loaded from src')

    except:
        print('no module src')


# TODO: Set relative path if needed (doesnt work right now but maybe for server)
# datapath = sys.path[len(sys.path) - 1] + "/data/"
datapath = 'C:\\Users\\langh\\Individual_project\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

#### CRUTS ####
print("Processing CRUTS")

vars = ['cld', 'dtr', 'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']
coords = ['lon', 'lat', 'month', 'year']

path_cruts = rawpath + 'CRUTS/'

path_var = path_cruts + 'cru_ts4.06.1901.2021.tmp.dat.nc/cru_ts4.06.1901.2021.tmp.dat.nc'
path_var_save = interrimpath + 'cruts_var/' + 'cruts_' + 'tmp' + '.csv'

df_cruts_out = xr.open_dataset(path_var).to_dataframe().reset_index()
df_cruts_out = df_cruts_out.dropna()
df_cruts_out = df_cruts_out[df_cruts_out["time"] < '2017-01-16']
df_cruts_out = df_cruts_out[df_cruts_out["time"] > '1980-01-16']
df_cruts_out['month'] = pd.DatetimeIndex(df_cruts_out['time']).month
df_cruts_out['year'] = pd.DatetimeIndex(df_cruts_out['time']).year
mainout = coords + ['tmp']
df_cruts_out = df_cruts_out[mainout]
df_cruts_out.set_index(coords, inplace=True)

# df_cruts_out.to_csv(path_var_save)

print('first done!')

for var in vars:
    print(var)
    path_var = path_cruts + 'cru_ts4.06.1901.2021.' + var + '.dat.nc\\cru_ts4.06.1901.2021.' + var + '.dat.nc'
    path_var_save = interrimpath + 'cruts_var/' + 'cruts_' + var + '.csv'
    print(path_var)
    df_var = xr.open_dataset(path_var).to_dataframe().reset_index()
    df_var = df_var.dropna()
    df_var = df_var[df_var["time"] > '1980-01-16']
    df_var = df_var[df_var["time"] < '2017-01-16']
    df_var['month'] = pd.DatetimeIndex(df_var['time']).month
    df_var['year'] = pd.DatetimeIndex(df_var['time']).year
    var_out = coords + [var]
    df_var = df_var[var_out]
    df_var.to_csv(path_var_save)
    df_var.set_index(coords, inplace=True)
    df_cruts_out = df_cruts_out.join(df_var, how='outer')



df_cruts_out = df_cruts_out.reset_index()
df_cruts_out['wet'] = df_cruts_out['wet'].apply(ut.timeToDays)
df_cruts_out['frs'] = df_cruts_out['frs'].apply(ut.timeToDays)

df_cruts_out.to_csv(interrimpath + 'df_cruts_mth.csv')

print('Done!')

del df_cruts_out, df_var
