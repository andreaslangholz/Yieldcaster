from re import X
import intake
import xarray as xr
import proplot as plot
import matplotlib.pyplot as plt
import pandas as pd
import src.utils as ut
import numpy as np

datapath = "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\"
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"
cmippath = interrimpath + 'cmip/'

# CRUTS vars:
# vars = ['cld', 'dtr', 'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']
cruts_to_cmip = {
    'cld' : 'clt',      # cloud cover
    'tmp' : 'tas',      # mean temperature
    'tmn' : 'tasmin',   # min temperatu
    'tmx' : 'tasmax',   # max temp
    'frs': 'fd',        # days with min temp < 0
    'wet': 'r10',       # days with +10 mm rain
    'pre': 'pr',         # precipitation
    'vap': 'prw'          # vapor pressure
}

vars_cruts = ['tmp','tmn', 'tmx', 'vap', 'pre', 'cld']
vars_cmip = []

for v in vars_cruts: vars_cmip = vars_cmip + [cruts_to_cmip[v]]

# Pangeo dataframe of all cmip-6 entries
url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"  # open the catalog

datastore = intake.open_esm_datastore(url)
datastore.df.columns

models = datastore.search(
    source_id = 'HadGEM3-GC31-MM',
    experiment_id= "ssp585",
    table_id= 'Amon',
    member_id='r1i1p1f3',
    variable_id = vars_cmip
)

datasets = models.to_dataset_dict()
keys = list(datasets.keys())
keys
ds = datasets[keys[0]]
lon = ds.lon
ds['lon'] = np.where(lon > 180, lon - 360, lon)

def round_coords(coord):
    x_abs = abs(coord)
    dif = x_abs - np.round(x_abs.values)
    new_dec = np.where(dif < 0, 0.75, 0.25)
    sign = np.where(coord < 0, -1, 1)
    x_out = sign * (np.floor(x_abs.values) + new_dec)
    return x_out


import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    x = array[idx]
    if x > value:
        y = array[idx - 1]
    else:
        y = array[idx + 1]
    return x, y

def euclid_dist(x1, y1, x2, y2):
    out = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return out

def convert(array_lon, array_lat, lon, lat):
    lon1, lon2 = find_nearest(array_lon, lon)
    lat1, lat2 = find_nearest(array_lat, lat)
    ar = np.array([[lon1, lat1, euclid_dist(lon1, lat1, lon, lat)],
        [lon1, lat2, euclid_dist(lon1, lat2, lon, lat)],
        [lon2, lat1, euclid_dist(lon2, lat1, lon, lat)],
        [lon2, lat2, euclid_dist(lon2, lat2, lon, lat)]])
    ara = ar[ar[:,2].argmin(),:]
    return ara[0], ara[1]

#TODO: Convert and match normal variables to clim models 
yieldmask = pd.read_csv(interrimpath + 'yieldmask.csv', index_col=0)

lamb_con = lambda x,y: convert(df_30_sub['lon'], df_30_sub['lat'], x, y)


df_30 = ds.where(ds['time.year'] == 2030, drop=True).to_dataframe().reset_index()
df_30 = df_30[vars_cmip + ['lon', 'lat', 'time']]

yieldmask['tuple'] = yieldmask.apply(lambda x: lamb_con(x['lon'], x['lat']), axis=1)
yieldmask['lon2'] = yieldmask['tuple'].apply(lambda t: t[0])
yieldmask['lat2'] = yieldmask['tuple'].apply(lambda t: t[1])

yieldmask
#TODO: merge with new coords
from datetime import datetime as dt

def convert_to_dt(x):
    return dt.strptime(str(x), '%Y-%m-%d %H:%M:%S')

df_30['datetime']=df_30['time'].apply(convert_to_dt)
df_30['month'] = pd.DatetimeIndex(df_30['datetime']).month
df_30['year'] = pd.DatetimeIndex(df_30['datetime']).year
df_30 = df_30[vars_cmip + ['lon', 'lat', 'year', 'month']]

df_30.to_csv(cmippath + 'hadgem_30.csv')

df_30[['prw', 'pr', 'clt']]
df_yield = pd.read_csv()

df = datasets[keys[0]].to_dataframe().reset_index()

