""" 
Data processing script
Objective: Detrends the future simulated featuresets 
Input: Processed simulated featuresets, original featureset to extrapolate trending line from
Output: Detrended matching featuresets
"""


import os
import pandas as pd
import multiprocessing as mp
from multiprocessing import freeze_support
from functools import partial
from sklearn.linear_model import LinearRegression
import numpy as np

def detrend(i, x, g):
    x1 = x[(x.lon == g.loc[i].lon) & (x.lat == g.loc[i].lat)].dropna()
    copies = 2101 - 1982
    out = pd.DataFrame(np.repeat(g.iloc[[i]].values,
                                    copies,
                                    axis=0),
                                    columns=g.columns).reset_index(drop=True)
    out['year'] = range(1982, 2101)
    linr = LinearRegression()
    vars = ['cld', 'tmp', 'tmn', 'pre', 'tmx', 'vap', 'heat','heat6mth']
    for var in vars:
        for c in range(1, 13):
            v = var + str(c)
            linr.fit( x1[['year']], x1[[v]])
            out[v] = linr.predict(out[['year']])
            if c not in ['heat', 'heat6mth']:
                v2 = v + '_roll5'
                linr.fit(x1[['year']], x1[[v2]])
                out[v2] = linr.predict(out[['year']])
    return out

def main():    
    current_directory = os.getcwd()
    datapath = current_directory + "/data/"
    process_path = datapath + "processed/"

    df_model = pd.read_csv(process_path + 'df_train_features_cruts.csv', index_col=0)
    g = df_model.groupby(['lon', 'lat']).size().reset_index().drop(0, axis=1)
    x = df_model
    x = x.sort_values(['lon', 'lat', 'year'], axis= 0)

    df_out = pd.DataFrame()
    num_pix = len(g)
    all_pix = [pix for pix in range(0,num_pix)]
    pool = mp.Pool(mp.cpu_count() - 3)
    results = pool.map(partial(detrend, x=x, g=g), all_pix)
    pool.close()
    print(results)

    df_out = results[0]
    for i in range(1, len(all_pix)):
        df_out = pd.concat([df_out, results[i]], axis=0, sort=False)

    df_out.to_csv(process_path + 'df_features_cruts_trends')

if __name__ == '__main__':
    print('running as module')
    freeze_support()
    main()
