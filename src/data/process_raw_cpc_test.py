from os.path import exists
from unittest import result
import xarray as xr
import pandas as pd
import multiprocessing as mp

def process_cpc(year):
    print(year, 'start')
    datapath = "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\"
    rawpath = datapath + "raw/"
    interrimpath = datapath + "interrim/"
    path_cpc = rawpath + "cpc/tmax."
    yieldmask = pd.read_csv(interrimpath + 'yieldmask.csv', index_col=0)
    df_year = xr.open_dataset(path_cpc + str(year) + '.nc').to_dataframe().reset_index()
    df_year['month'] = pd.DatetimeIndex(df_year['time']).month
    df_year['year'] = str(year)
    thresholds = {"wheat": 30, "maizerice": 35, "soy": 39}
    minzero = lambda x : max(0, x)
    recenter_lon = lambda lon : lon - 360 if (lon > 180) else lon
    for crop in ["wheat", "maizerice", "soy"]:
        df_year[crop + '_mth'] = (df_year['tmax'] - thresholds[crop]).transform(minzero)    
    df_year = df_year.groupby(["lat", "lon", "month", "year"])["wheat_mth", "maizerice_mth", "soy_mth"].sum().reset_index()
    df_year['lon'] = df_year['lon'].apply(recenter_lon)
    df_year = df_year.merge(yieldmask, on = ['lon', 'lat'], how = 'inner')
    print(year, 'end')
    return df_year

if __name__ == '__main__':
    years = range(1981, 2017)
    pool = mp.Pool(mp.cpu_count() - 1)
    results = pool.map(process_cpc, [year for year in years])
    pool.close()
    print(results)
    
    df_tmax_out = results[0]
    for i in (len(results) - 1):
        df_tmax_out = pd.concat([df_tmax_out, results[i]], sort=False)

    datapath = "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\"
    interrimpath = datapath + "interrim/"
    df_tmax_out.to_csv(interrimpath + 'cpc_interrim.csv')


