import intake
import xarray as xr
import proplot as plot
import matplotlib.pyplot as plt
import pandas as pd

# CRUTS vars:
# vars = ['cld', 'dtr', 'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']
cruts_to_cmip = {
    'cld' : 'clt',      # cloud cover
    'tmp' : 'tas',      # mean temperature
    'tmn' : 'tasmin',   # min temperatu
    'tmx' : 'tasmax',   # max temp
    'frs': 'fd',        # days with min temp < 0
    'wet': 'r10',       # days with +10 mm rain
    'pet': 'x',         # potential evapotranspiration
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
    variable_id= vars_cmip
)

datasets = models.to_dataset_dict()
keys = list(datasets.keys())

datasets[keys[0]]

