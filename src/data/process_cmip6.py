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
    'dtr': 'x',         # diurnal temp mean
    'pet': 'x',         # potential evapotranspiration
    'pre': 'pr',         # precipitation
    'vap': 'prw'          # vapor pressure
}

vars_cruts = ['tmp','tmn', 'tmx', 'vap', 'wet', 'cld', 'frs', 'pre']

vars_cmip = []
for var in vars_cruts:
    vars_cmip.append(cruts_to_cmip[var])

# Pangeo dataframe of all
url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"  # open the catalog
datastore = intake.open_esm_datastore(url)

models = datastore.search(
    source_id = 'HadGEM3-GC31-MM',
    experiment_id=["ssp585"],
    table_id='Amon',
    variable_id=vars_cmip)

datasets = models.to_dataset_dict()
keys = list(datasets.keys())

dfset = datasets[keys[0]].to_dataframe().reset_index()
dfset.columns

fig, ax = plot.subplots(axwidth=4.5, tight=True,
                        proj='robin', proj_kw={'lon_0': 180}, )
# format options
ax.format(land=False, coast=True, innerborders=True, borders=True,
          labels=True, gridlinewidth=0, )

dfs = dfset[['lat', 'lon', 'tas', 'time']]
dfst = dfs[dfs['time'] == dfs['time'][len(dfs) - 1]]
dfset.describe()
