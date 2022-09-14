""" 
Data processing script: Process the GDHY files
Objective: Combine and standardise the raw GDHY files into 1 dataframe
Input: directories with netcdf files for each crop-year in structure: crop/yield_year.netcdf
Output: csv files with dataframes for each crop and 1 combined file, 1 yieldmask file with individual entries for all non-zero pixels
"""

if __name__ == '__main__':
    print('running process_raw_gdhy.py')

import pandas as pd
from pkg_resources import yield_lines
import xarray as xr
from os.path import exists
import numpy as np
# import src.utils as ut

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

datapath = 'C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\'
rawpath = datapath + "raw/"
interrimpath = datapath + "interrim/"

crops = ['wheat_winter', 'wheat_spring', 'soy', 'maize_major', 'rice_major']
years = range(1981, 2017)

for crop in crops:
    print(crop)
    croppath = rawpath + 'gdhy\\' + crop + '\\'
    df_yield_out = pd.DataFrame(columns=['lon', 'lat', 'year', 'yield'])

    for year in years:
        print(crop, year)
        croppath_year = croppath + 'yield_' + str(year) + '.nc4'

        df_yield = xr.open_dataset(croppath_year).to_dataframe().reset_index()
        df_yield = df_yield.dropna().reset_index(drop=True)
        df_yield['year'] = year
        df_yield['lon'] = df_yield['lon'].apply(ut.recenter_lon)
        df_yield.rename(columns={'var': 'yield'}, inplace=True)
        df_yield_out = pd.concat([df_yield_out, df_yield])

    if crop == 'maize_major':
        crop = 'maize'
    elif crop == 'rice_major':
        crop = 'rice'

    df_yield_out.to_csv(interrimpath + 'yield\\' + 'df_' + crop + '_yield.csv')

# Make combined crops set:
df_yield_out = pd.DataFrame(columns = ['lon', 'lat', 'year', 'yield', 'harvest', 'crop'])
crops = ['wheat_winter', 'wheat_spring', 'soy', 'maize', 'rice']

for crop in crops:
    df_yield = pd.read_csv(interrimpath + 'yield\\' + 'df_' + crop + '_yield.csv', index_col=0)
    df_yield['crop'] = crop
    
    if crop == 'wheat_spring' or crop == 'wheat_winter':
        c = 'wheat'
    else:
        c = crop
    df_harvest = pd.read_csv(interrimpath + "harvest/df_" + c +"_har.csv", index_col=0)
    df_yield = pd.merge(df_yield, df_harvest, on=['lon', 'lat'], how = 'inner')
    df_yield_out = pd.concat([df_yield_out, df_yield])

df_yield_out.reset_index(drop = True, inplace=True)
df_yield_out.to_csv(interrimpath + 'yield\\' + 'df_all_yield.csv')

# Make yieldmask:
crops = ['wheat_winter', 'wheat_spring', 'soy', 'maize', 'rice']

yield_mask = pd.DataFrame(columns=['lon', 'lat'])

for crop in crops:
    print(crop)
    df_yield = pd.read_csv(interrimpath + 'yield\\' + 'df_' + crop + '_yield.csv', index_col=0)
    df_yield = df_yield.groupby(['lon', 'lat']).count().reset_index()
    df_yield = df_yield[['lat', 'lon']]
    print(len(df_yield))
    yield_mask = yield_mask.merge(df_yield, on = ['lat', 'lon'], how='outer')

yield_mask['pixel_id']  = np.arange(len(yield_mask))

yield_mask.to_csv(interrimpath + 'yieldmask.csv')


