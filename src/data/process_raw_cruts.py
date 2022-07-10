if __name__ == '__main__':
    print('running process_raw_cruts.py')

import pandas as pd
import xarray as xr
from os.path import exists
import gc

import sys
# sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append("..")
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

datapath = 'C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"
path_cruts = rawpath + 'cruts/'
vars = ['tmp','tmn', 'tmx', 'vap', 'pre']

zip = False

try:
    zip = sys.argv[1]
except:
    print('No unzipping')

if zip:
    print("Unzipping")
    for var in vars:
        path_var = path_cruts + 'gz/' + 'cru_ts4.05.1901.2020.'+ var +'.dat.nc.gz'
        path_var_out = path_cruts + 'cruts_' + var +'.nc'
        ut.gunzip(path_var, path_var_out)

#### CRUTS ####
print("Processing CRUTS")

coords = ['lon', 'lat', 'month', 'year']
yieldmask = pd.read_csv(interrimpath + 'yieldmask.csv', index_col=0)

var = 'cld'

path_var = path_cruts + 'cruts_' + var +'.nc'
path_var_save = interrimpath + 'cruts_var/' + 'cruts_' + var + '.csv'

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

print('first done!')

df_cruts_out = df_var
df_cruts_out.set_index(coords, inplace=True)

for var in vars:
    print(var)
    path_var = path_cruts + 'cruts_' + var +'.nc'
    path_var_save = interrimpath + 'cruts_var/' + 'cruts_' + var + '.csv'

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
df_cruts_out.to_csv(interrimpath + 'df_cruts_mth.csv')

print('Done!')
df_cruts_out
del df_cruts_out
