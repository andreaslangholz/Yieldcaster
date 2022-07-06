import os.path
if __name__ == '__main__': print('running process_raw_spei.py')

import pandas as pd
import xarray as xr

# TODO: Set relative path if needed (doesnt work right now but maybe for server)
# datapath = sys.path[len(sys.path) - 1] + "/data/"
datapath = 'C:\\Users\\langh\\Individual_project\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

### SPEI

print("Processing SPEI")
path_spei = rawpath + "SPEI/SPEI_12_Amon_EC-EARTH3-HR_rcp85_r1i1p1.nc"

df_spei = xr.open_dataset(path_spei).to_dataframe().reset_index()

df_spei['month'] = pd.DatetimeIndex(df_spei['time']).month
df_spei['year'] = pd.DatetimeIndex(df_spei['time']).year
df_spei = df_spei.dropna()
df_spei = df_spei.sort_values(by=['lon', 'lat', 'year', 'month'])
df_spei['spei_9mths'] = df_spei['SPEI'].rolling(9).sum()

#TODO: sort by landmask

df_spei.drop('time', axis=1, inplace = True)
df_spei_out = df_spei

df_spei_out.to_csv(interrimpath + 'df_spei_mth_all.csv')

df_spei_out = df_spei_out[(df_spei_out["year"] > 1980) & (df_spei_out["year"] < 2017)]

df_spei_out.to_csv(interrimpath + 'df_spei_mth_8017.csv')

print('Done!')
del df_spei
