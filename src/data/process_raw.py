import sys
import pandas as pd
import xarray as xr
from src.utils import maximumZero

datapath = sys.path[len(sys.path) - 1] + "/data/"
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

## CPC TMAX ###
print("Processing CPC")

path_cpc = rawpath + "CPC Daily/tmax."
thresholds = {"wheat": 30, "maize": 35, "rice": 35, "soy": 39}

df_tmax = pd.DataFrame(columns=['lat', 'lon', 'time', 'month', 'year', 'tmax'])

for year in range(1980, 2017):
    print(year)
    df_year = xr.open_dataset(path_cpc + str(year) + '.nc').to_dataframe().reset_index()
    df_year['month'] = pd.DatetimeIndex(df_year['time']).month
    df_year['year'] = pd.DatetimeIndex(df_year['time']).year
    df_year = df_year.dropna()
    df_tmax = pd.concat([df_tmax, df_year], sort=False)

for crop in ["wheat", "maize", "rice", "soy"]:
    df_mths_crop = df_tmax.copy()
    max_ = crop + 'max'
    df_mths_crop[max_] = (df_mths_crop['tmax'] - thresholds[crop]).transform(maximumZero)
    df_mths_crop = df_mths_crop.groupby(["lat", "lon", "month", "year"])[max_] \
        .sum().reset_index()
    df_mths_crop.to_csv(interrimpath + 'df_heat_' + crop + '.csv')
    del df_mths_crop

del df_tmax, df_year

### SPEI
# TODO: Check consistency (e.g. are there records with NaNs in certain months/years?)

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

#### CRUTS ####
print("Processing CRUTS")

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

df_cruts_out.set_index(['lat', 'lon', 'month', 'year'], inplace=True)

for var in vars:
    print(var)
    path_var = path_cruts + 'cru_ts4.06.1901.2021.' + var + '.dat.nc\\cru_ts4.06.1901.2021.' + var + '.dat.nc'
    df_var = xr.open_dataset(path_var).to_dataframe().reset_index()
    df_var = df_var[df_var["time"] > '1980-01-16']
    df_var = df_var[df_var["time"] < '2017-01-16']
    df_var['month'] = pd.DatetimeIndex(df_var['time']).month
    df_var['year'] = pd.DatetimeIndex(df_var['time']).year
    var_out = coords + [var]
    df_var = df_var[var_out]
    path_var_save = interrimpath + 'cruts_var/' + 'cruts_' + var + '.csv'
    df_var.to_csv(path_var_save)
    df_var.set_index(['lat', 'lon', 'month', 'year'], inplace=True)
    df_cruts_out = df_cruts_out.join(df_var, how='outer', on=coords)

df_cruts_out['wet'] = df_cruts_out['wet'].dt.days
df_cruts_out['frs'] = df_cruts_out['frs'].dt.days

df_cruts_out.to_csv(interrimpath + 'df_cruts_mth.csv')

del df_cruts_out, df_var

#### Combine variables and flatten the structure ####
df_spei = pd.read_csv(interrimpath + 'spei_processed.csv', index_col=0)
df_cruts = pd.read_csv(interrimpath + 'cruts_processed.csv', index_col=0)

df_spei.set_index(['lat', 'lon', 'month', 'year'], inplace=True)
df_cruts.set_index(['lat', 'lon', 'month', 'year'], inplace=True)

df_var = df_spei.join(df_cruts, how='outer')
df_var.reset_index()

df_var.to_csv(interrimpath + 'df_var_common_mth.csv')
del df_var

#### GDHY ####
print("Processing GDHY")
path_yield = rawpath + 'gdhy_v1.3/'
crops = ['maize', 'soybean', 'wheat', 'rice']

for crop in crops:
    print(crop)
    path = path_yield + crop + '/'
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
    df_yield_out.to_csv(interrimpath + 'df_' + crop + '_yield.csv')

del df_yield_out, df_yield_year
