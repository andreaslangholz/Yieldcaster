if __name__ == '__main__':
    print('running process_cpc_raw.py')    

from os.path import exists
import xarray as xr
import pandas as pd
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
# datapath = sys.path[len(sys.path) - 1] + "/data/"
datapath = "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\"
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

## CPC TMAX ###
print("Processing CPC")

path_cpc = rawpath + "cpc/tmax."
thresholds = {"wheat": 30, "maizerice": 35, "soy": 39}

# If full merge breaks, then start from here

df_tmax = xr.open_dataset(path_cpc + str(1981) +
                          '.nc').to_dataframe().reset_index()

for crop in ["wheat", "maizerice", "soy"]:
    df_tmax[crop + '_mth'] = (df_tmax['tmax'] -
                              thresholds[crop]).transform(ut.maximumZero)

df_tmax['month'] = pd.DatetimeIndex(df_tmax['time']).month
df_tmax['year'] = pd.DatetimeIndex(df_tmax['time']).year
df_tmax_out = df_tmax.groupby(["lat", "lon", "month", "year"])["wheat_mth", "maizerice_mth", "soy_mth"] \
    .sum().reset_index()

df_tmax_out['lon'] = df_tmax_out['lon'].apply(ut.recenter_lon)
yieldmask = pd.read_csv(interrimpath + 'yieldmask.csv', index_col=0)
df_tmax_out = df_tmax_out.merge(yieldmask, on = ['lon', 'lat'], how = 'inner')

print('first!')

for year in range(1982, 2017):
    print(year)
    df_year = xr.open_dataset(
        path_cpc + str(year) + '.nc').to_dataframe().reset_index()
    df_year['month'] = pd.DatetimeIndex(df_year['time']).month
    df_year['year'] = pd.DatetimeIndex(df_year['time']).year

    for crop in ["wheat", "maizerice", "soy"]:
        df_year[crop + '_mth'] = (df_year['tmax'] -
                                  thresholds[crop]).transform(ut.maximumZero)

    df_year = df_year.groupby(["lat", "lon", "month", "year"])[["wheat_mth", "maizerice_mth", "soy_mth"]] \
        .sum().reset_index()

    df_year['lon'] = df_year['lon'].apply(ut.recenter_lon)
    df_year = df_year.merge(yieldmask, on = ['lon', 'lat'], how = 'inner')
    df_tmax_out = pd.concat([df_tmax_out, df_year], sort=False)
    df_tmax_out.to_csv(interrimpath + 'cpc_interrim.csv')

df_tmax_out = df_tmax_out.sort_values(by=['lon', 'lat', 'year', 'month'])
for crop in ["wheat", "maizerice", "soy"]:
    df_tmax_out[crop + '_6mths'] = df_tmax_out[crop + '_mth'].rolling(6).sum()

df_tmax_out['lon'] = df_tmax_out['lon'].apply(ut.recenter_lon)
df_tmax_out.to_csv(interrimpath + 'df_heat_mth.csv')

print('fucking done!')
