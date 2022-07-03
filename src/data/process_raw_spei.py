import os.path
if __name__ == '__main__': print('running process_raw_spei.py')

import pandas as pd
import xarray as xr

# TODO: Set relative path if needed (doesnt work right now but maybe for server)
# datapath = sys.path[len(sys.path) - 1] + "/data/"
datapath = 'C:\\Users\\langh\\OneDrive\\Documents\\Imperial\\Individual_project\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

### SPEI

print("Processing SPEI")
path_spei = rawpath + "SPEI/spei01.nc"

df_spei = xr.open_dataset(path_spei).to_dataframe().reset_index()
df_spei = df_spei[df_spei["time"] > '1980-01-16']
df_spei = df_spei[df_spei["time"] < '2017-01-16']
df_spei['month'] = pd.DatetimeIndex(df_spei['time']).month
df_spei['year'] = pd.DatetimeIndex(df_spei['time']).year
df_spei = df_spei.dropna()
df_spei = df_spei.sort_values(by=['lon', 'lat', 'year', 'month'])
df_spei['spei_9mths'] = df_spei['spei'].rolling(9).sum()

df_spei.to_csv(interrimpath + 'df_spei_mth.csv')

del df_spei
