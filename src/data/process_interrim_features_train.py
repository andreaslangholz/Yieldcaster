""" 
Data processing script: training features
Objective: Make combined datasets of historically observed features from the cruts set and the heat data
Input: datasets for CRUTS and Heat features
Output: 
"""

import os
import pandas as pd
from src.h_utils import df_mth_to_year

current_directory = os.getcwd()
datapath = current_directory + "/data/"
int_path = datapath + "interrim/"
process_path = datapath + "processed/"

df_cruts = pd.read_csv(int_path + 'df_cruts_mth.csv', index_col=0)
df_heat = pd.read_csv(int_path + "df_heat_mth.csv", index_col=0)

# OBS: leave out if using crop specific heat threshold (already calculated in pre-processing)
# df_heat['heat'] = df_heat['maizerice_mth']
# df_heat['heat6mth'] = df_heat['maizerice_6mths']
# df_heat = df_heat[['lon', 'lat', 'year', 'month', 'heat',  'heat6mth']]

# Convert to monthly set
df_cruts_mth = df_mth_to_year(df_cruts)

# Calculate rolling averages
for col in df_cruts_mth.columns:
    if col not in ['lon', 'lat', 'year']:
        print(col)      
        df_cruts_mth[col + '_roll5'] = df_cruts_mth.groupby(['lon', 'lat'], as_index=False)[[col]].rolling(5, min_periods=1).mean().reset_index(0, drop=True)[col]

df_heat_mth = df_mth_to_year(df_heat)

df_out = df_cruts_mth.merge(df_heat_mth, on=['lon', 'lat', 'year'])
df_out.to_csv(process_path + 'df_train_features_cruts.csv')
