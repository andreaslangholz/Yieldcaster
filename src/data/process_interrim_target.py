""" 
Data processing script: Format target dataset
Objective: Create target datasets with normal and detrended features
Input: Interrim yield data
Output: 2 dataframes with normal and detrended targets
"""

import pandas as pd
import os    
from scipy import signal

current_directory = os.getcwd()
datapath = current_directory + "/data/"
int_path = datapath + "interrim/"
process_path = datapath + "processed/"

df_yield = pd.read_csv(int_path + 'yield/' + 'df_all_yield.csv', index_col=0)
df_yield = df_yield[(df_yield.year >= 1982) & (df_yield.year <= 2015)]
df_yield = df_yield.dropna()

df_yield.to_csv(process_path + "df_train_target.csv")
df_yield = pd.read_csv(process_path + "df_train_target.csv", index_col = 0)

# Detrending
df_g = df_yield.groupby(['lon', 'lat', 'crop']).size().reset_index()

df_trend = pd.DataFrame()

df_yield = df_yield.sort_values(['lon', 'lat', 'crop', 'year'])
df_yield['yield'].describe()

df_trend = pd.DataFrame()
for i in range(0, len(df_g)):
    coord = df_g.loc[i]
    df_t = df_yield[(df_yield.lon == coord.lon) & (df_yield.lat == coord.lat) & (df_yield.crop == coord.crop)]
    df_t['yield'] = signal.detrend(df_t['yield'])
    df_trend = pd.concat([df_trend, df_t], axis=0)

df_trend = df_trend.reset_index(drop=True)

df_trend.to_csv(process_path + "df_train_target_trend.csv")
