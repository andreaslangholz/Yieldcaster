if __name__ == '__main__': print('running process_cpc_raw.py')

import sys
sys.path.append("..")# sys.path.append("..") # Adds higher directory to python modules path.

import pandas as pd
import xarray as xr
import utils as ut

# TODO: Set relative path if needed (doesnt work right now but maybe for server)
# datapath = sys.path[len(sys.path) - 1] + "/data/"
datapath = 'C:\\Users\\langh\\OneDrive\\Documents\\Imperial\\Individual_project\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

## CPC TMAX ###
print("Processing CPC")

path_cpc = rawpath + "CPC Daily/tmax."
thresholds = {"wheat": 30, "maize": 35, "rice": 35, "soy": 39}

df_tmax = pd.DataFrame(columns=['lat', 'lon', 'time', 'month', 'year', 'tmax'])

# If full merge breaks, then start from here
df_tmax = pd.read_csv(interrimpath + 'cpc_interrim.csv', index_col=0)

start_year = 1980 if len(df_tmax) == 0 else df_tmax['year'].max() + 1
print(start_year)

for year in range(start_year, 2017):
    print(year)
    df_year = xr.open_dataset(path_cpc + str(year) + '.nc').to_dataframe().reset_index()
    df_year['month'] = pd.DatetimeIndex(df_year['time']).month
    df_year['year'] = pd.DatetimeIndex(df_year['time']).year
    df_year = df_year.dropna()
    df_tmax = pd.concat([df_tmax, df_year], sort=False)
    df_tmax.to_csv(interrimpath + 'cpc_interrim.csv')
    print(df_tmax['year'].unique())

print('Summing days')
for crop in ["wheat", "maize", "rice", "soy"]:
    df_mths_crop = df_tmax.copy()
    max_ = crop + 'max'
    df_mths_crop[max_] = (df_mths_crop['tmax'] - thresholds[crop]).transform(ut.maximumZero)
    df_mths_crop = df_mths_crop.groupby(["lat", "lon", "month", "year"])[max_] \
        .sum().reset_index()
    df_mths_crop.to_csv(interrimpath + 'df_heat_' + crop + '.csv')
    del df_mths_crop

del df_tmax, df_year

print('test')
