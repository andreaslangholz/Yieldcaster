import os.path

if __name__ == '__main__': print('running process_raw.py')

import sys
sys.path.append("..")# sys.path.append("..") # Adds higher directory to python modules path.

import pandas as pd
import xarray as xr
from os.path import exists

# TODO: Set relative path if needed (doesnt work right now but maybe for server)
# datapath = sys.path[len(sys.path) - 1] + "/data/"
datapath = 'C:\\Users\\langh\\OneDrive\\Documents\\Imperial\\Individual_project\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

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
