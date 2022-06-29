import sys
import pandas as pd
import xarray as xr
from src.utils import maximumZero, timeToDays

datapath = sys.path[len(sys.path)-1] + "/data/"
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

## CPC TMAX ###
path_cpc = rawpath + "CPC Daily/tmax."

thresholds = {"wheat": 30, "maize": 35, "rice": 35, "soy": 39}
df_tmax = xr.open_dataset(path_cpc + "1980.nc").to_dataframe().reset_index()
df_tmax['month'] = pd.DatetimeIndex(df_tmax['time']).month
df_tmax['year'] = pd.DatetimeIndex(df_tmax['time']).year

for crop in ["wheat", "maize", "rice", "soy"]:
    df_tmax[crop + 'dif'] = df_tmax['tmax'] - thresholds[crop]
    df_tmax[crop + 'max'] = df_tmax[crop + 'dif'].transform(maximumZero)

df_tmax_out = df_tmax.groupby(["lat", "lon", "month", "year"])["wheatmax", "maizemax", "ricemax", "soymax"] \
    .sum().reset_index()

for year in range(1981, 2017):
    print(year)
    df_year = xr.open_dataset(path_cpc + str(year) + '.nc').to_dataframe().reset_index()
    df_year['month'] = pd.DatetimeIndex(df_year['time']).month
    df_year['year'] = pd.DatetimeIndex(df_year['time']).year
    df_year = df_year.dropna()
    for crop in ["wheat", "maize", "rice", "soy"]:
        df_year[crop + 'dif'] = df_year['tmax'] - thresholds[crop]
        df_year[crop + 'max'] = df_year[crop + 'dif'].transform(maximumZero)

    df_tmax_grouped = df_year.groupby(["lat", "lon", "month", "year"])["wheatmax", "maizemax", "ricemax", "soymax"] \
        .sum().reset_index()
    df_tmax_out = pd.concat([df_tmax_out, df_tmax_grouped], sort=False)

df_tmax_out.to_csv(interrimpath + 'cpc_tmax.csv')

### SPEI
# TODO: Check consistency (e.g. are there records with NaNs in certain months/years?)
path_spei = rawpath + "SPEI/spei01.nc"
df_spei = xr.open_dataset(path_spei).to_dataframe().reset_index()
df_spei = df_spei[df_spei["time"] > '1980-01-16']
df_spei = df_spei[df_spei["time"] < '2017-01-16']
df_spei['month'] = pd.DatetimeIndex(df_spei['time']).month
df_spei['year'] = pd.DatetimeIndex(df_spei['time']).year
df_spei = df_spei.dropna()
df_spei = df_spei.sort_values(by=['lon', 'lat', 'year', 'month'])
df_spei['rolling'] = df_spei['spei'].rolling(9).sum()
df_spei.to_csv(interrimpath + 'spei_processed.csv')

#### CRUTS ####
vars = ['cld', 'dtr', 'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']
coords = ['lon', 'lat', 'month', 'year']

path_cruts = rawpath + 'CRUTS/'
path_var = path_cruts + 'cru_ts4.06.1901.2021.tmp.dat.nc/cru_ts4.06.1901.2021.tmp.dat.nc'

df_cruts_out = xr.open_dataset(path_var).to_dataframe().reset_index()
df_cruts_out = df_cruts_out[df_cruts_out["time"] > '1980-01-16']
df_cruts_out = df_cruts_out[df_cruts_out["time"] < '2017-01-16']
df_cruts_out['month'] = pd.DatetimeIndex(df_cruts_out['time']).month
df_cruts_out['year'] = pd.DatetimeIndex(df_cruts_out['time']).year
mainout = coords + ['tmp']
df_cruts_out = df_cruts_out[mainout]

for var in vars:
    path_var = path_cruts + 'cru_ts4.06.1901.2021.' + var + '.dat.nc\\cru_ts4.06.1901.2021.' + var +'.dat.nc'
    df_var = xr.open_dataset(path_var).to_dataframe().reset_index()
    df_var = df_var[df_var["time"] > '1980-01-16']
    df_var = df_var[df_var["time"] < '2017-01-16']
    df_var['month'] = pd.DatetimeIndex(dfvar['time']).month
    df_var['year'] = pd.DatetimeIndex(dfvar['time']).year
    df_out = coords + [var]
    df_var = dfvar[df_out]
    path_var_save = interrimpath + 'cruts_var/' + 'cruts_' + var + '.csv'
    df_var.to_csv(path_var_save)
    df_cruts_out = df_cruts_out.merge(df_var, how='inner', on=coords)

df_cruts_out.to_csv(interrimpath + 'cruts_processed.csv')

####  YIELD ####
path_yield = rawpath + 'gdhy_v1.3\'
crops = ['maize', 'soybean', 'wheat', 'rice']

for crop in crops:
    path = path_yield + crop + '/'
    print('Path: '+path)
    df_yield_out = xr.open_dataset(path + 'yield_' + '1982' + '.nc4')
    df_yield_out = df_yield_out.to_dataframe().reset_index()
    df_yield_out = df_yield_out.rename(columns={"var": "yield"})
    df_yield_out['year'] = '1982'

    for year in range(1983, 2016):
        print(year)
        df_yield_year = xr.open_dataset(path + 'yield_' + str(year) + '.nc4').to_dataframe().reset_index()
        df_yield_year = df_yield_year.rename(columns={"var": "yield"})
        df_yield_year['year'] = str(year)
        df_yield_out = pd.concat([df_yield_out, df_yield_year], sort=False)
    df_yield_out.to_csv(interrimpath + crop + '_yield.csv')

