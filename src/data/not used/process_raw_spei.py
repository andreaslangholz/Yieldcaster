import os.path
if __name__ == '__main__': print('running process_raw_spei.py')

import pandas as pd
import xarray as xr
import src.utils as ut

# TODO: Set relative path if needed (doesnt work right now but maybe for server)
# datapath = sys.path[len(sys.path) - 1] + "/data/"
datapath = 'C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

### SPEI

print("Processing SPEI")
path_spei = rawpath + "spei/SPEI_12_Amon_EC-EARTH3-HR_rcp85_r1i1p1.nc"

yieldmask = pd.read_csv(interrimpath + 'yieldmask.csv', index_col=0)

df_spei = xr.open_dataset(path_spei).to_dataframe().reset_index()
df_spei = df_spei.dropna()
df_land = ut.subset(df_spei)
df_land = df_land[['lon', 'lat']]
df_land.to_csv(interrimpath + 'landmask.csv')
df_spei = df_spei.merge(yieldmask, on = ['lon', 'lat'], how = 'inner')
df_spei['month'] = pd.DatetimeIndex(df_spei['time']).month
df_spei['year'] = pd.DatetimeIndex(df_spei['time']).year
df_spei.drop('time', axis=1, inplace = True)
df_spei = df_spei.sort_values(by=['lon', 'lat', 'year', 'month'])
df_spei['spei_9mths'] = df_spei['SPEI'].rolling(9).sum()

df_spei_out = df_spei
df_spei_out.to_csv(interrimpath + 'df_spei_mth_all.csv')
df_spei_out = df_spei_out[(df_spei_out["year"] > 1980) & (df_spei_out["year"] < 2017)]
df_spei_out.to_csv(interrimpath + 'df_spei_mth_8017.csv')

print('Done!')
del df_spei
