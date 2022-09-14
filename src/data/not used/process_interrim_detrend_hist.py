""" 
Data processing script
Objective: Detrends the historical featuresets for both observed (CRUTS) and simulated average datasets
Input: Processed featuresets (observed and simulated)
Output: Detrended matching featuresets
"""

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal

current_directory = os.getcwd()
datapath = current_directory + "/data/"
int_path = datapath + "interrim/"
process_path = datapath + "processed/"

df_model = pd.read_csv(process_path + 'df_train_features_cruts.csv', index_col=0)
g = df_model.groupby(['lon', 'lat']).size().reset_index().drop(0, axis=1)
x = df_model
x = x.sort_values(['lon', 'lat', 'year'], axis= 0)

linr = LinearRegression()
vars = ['cld', 'tmp', 'tmn', 'pre', 'tmx', 'vap', 'heat','heat6mth']

df_out = pd.DataFrame()

# Detrend cruts
for i in range(1, len(g)):
    x1 = x[(x.lon == g.loc[i].lon) & (x.lat == g.loc[i].lat)].dropna()
    for var in vars:
        for c in range(1, 13):
            v = var + str(c)
            x1[v] = signal.detrend(x1[v])
            if var not in ['heat', 'heat6mth']:
                v2 = v + '_roll5'
                x1[v] = signal.detrend(x1[v2])
    df_out = pd.concat([df_out, x1], axis=0, sort=False)

df_out.to_csv(process_path + 'df_features_cruts_dt')

print('detrending historical simulations')
# Do the same for the simulated cruts

df_model = pd.read_csv(process_path + 'avg_historical_pred_cruts.csv', index_col=0)
g = df_model.groupby(['lon', 'lat']).size().reset_index().drop(0, axis=1)
x = df_model
x = x.sort_values(['lon', 'lat', 'year'], axis= 0)

linr = LinearRegression()
vars = ['cld', 'tmp', 'tmn', 'pre', 'tmx', 'vap', 'heat','heat6mth']

df_out = pd.DataFrame()
n_p = len(g)
for i in range(1, n_p):
    print(i, ' out of ', n_p)
    x1 = x[(x.lon == g.loc[i].lon) & (x.lat == g.loc[i].lat)].dropna()
    for var in vars:
        for c in range(1, 13):
            v = var + str(c)
            x1[v] = signal.detrend(x1[v])
            if var not in ['heat', 'heat6mth']:
                v2 = v + '_roll5'
                x1[v] = signal.detrend(x1[v2])
    df_out = pd.concat([df_out, x1], axis=0, sort=False)

df_out.to_csv(process_path + 'avg_historical_pred_cruts_dt.csv')
