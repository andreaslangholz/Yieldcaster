""" 
Data processing script
Objective: Standardises simulated featuresets, calculate new features and create model averages for each scenario
Input: Interrim features for each climate model X scenario
Output: final simulated featureset to predict model on
"""

from genericpath import exists
import os
import pandas as pd
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd)
import gc

from src.h_utils import df_mth_to_year

current_directory = os.getcwd()
datapath = current_directory + "/data/"
int_path = datapath + "interrim/"
process_path = datapath + "processed/"

models = ['MIROC6', 'ACCESS-ESM1-5', 'CanESM5'] #, 'EC-Earth3-Veg']

scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585'] # , '

# Process each model X scenario
for scenario in scenarios:
    for model in models:
        realpath = int_path + 'cmip/' + model + '_' + scenario + '_cruts.csv'
        outpath = process_path + model + '_' + scenario + '_pred_cruts.csv'
        if exists(outpath):
            print('skipping', model, scenario)
        else:
            print('processing', scenario, model)
            df_real = pd.read_csv(realpath, index_col=0)
            # Convert from Kelvin to Celcius
            df_real['tas']    = df_real['tas'] - 272.15
            df_real['tasmin'] = df_real['tasmin'] - 272.15
            df_real['tasmax'] = df_real['tasmax'] - 272.15
            try:
                df_real.drop('time', axis = 1, inplace = True)
            except:
                pass

            #Rename so it fits with Cruts naming   
            df_real.rename(columns = {
                'clt'       : 'cld' ,  # cloud cover
                'tas'       : 'tmp' ,  # mean temperature
                'tasmin'    : 'tmn',   # min temperatu
                'tasmax'    : 'tmx',   # max temp
                'pr'        : 'pre' ,     # precipitation
                'prw'       : 'vap' ,
                'heat_mth'  : 'heat'     
            }, inplace=True)

            vars = df_real.columns
            out = ['lon', 'lat','month', 'year', 'heat', 'heat6mth']
            roll_var = [x for x in vars if x not in out]
            df_real_mth = df_mth_to_year(df_real)
            # make rolling average 
            for var in roll_var:
                for i in range(1,13):
                    v = var + str(i)
                    df_real_mth[v + '_roll5'] = df_real_mth.groupby(['lon', 'lat'], as_index=False)[[v]].rolling(5, min_periods=1).mean().reset_index(0, drop=True)[v]
            df_real_mth.to_csv(outpath)

# Average climate model features across scenarios:
print('Making averages')
for scenario in scenarios:
    outpath = process_path + 'avg_' + scenario + '_pred_cruts.csv'
    if exists(outpath):
        print('done already - avg ', scenario)
    else:
        print('computing avg', scenario)
        df_ac = pd.read_csv(process_path + 'ACCESS-ESM1-5' + '_' + scenario + '_pred.csv', index_col=0)
        df_ca = pd.read_csv(process_path + 'CanESM5' + '_' + scenario + '_pred.csv', index_col=0)
        df_mi = pd.read_csv(process_path + 'MIROC6' + '_' + scenario + '_pred.csv', index_col=0)

        collected = gc.collect()
        print("Garbage collector: collected {} objects.".format(collected))
        
        df_ac.sort_values(['lon', 'lat', 'year'], inplace=True)
        df_ca.sort_values(['lon', 'lat', 'year'], inplace=True)
        df_mi.sort_values(['lon', 'lat', 'year'], inplace=True)
        
        df_ac.set_index(['lon', 'lat', 'year'], inplace = True)
        df_ca.set_index(['lon', 'lat', 'year'], inplace = True)
        df_mi.set_index(['lon', 'lat', 'year'], inplace = True)
        
        df_concat = pd.concat((df_ac, df_ca, df_mi))
        df_out = df_concat.groupby(df_concat.index).mean()
        df_out = df_out.reset_index()
        
        df_out['lon'] = df_out['index'].apply(lambda x: x[0])
        df_out['lat'] = df_out['index'].apply(lambda x: x[1])
        df_out['year'] = df_out['index'].apply(lambda x: x[2])
        df_out.drop('index', inplace = True, axis = 1)
    
        df_out.to_csv(outpath)

