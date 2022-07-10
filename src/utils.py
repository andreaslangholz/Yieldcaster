import torch
import pandas as pd
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
import re
import gzip

#print(__package__)
#print(__name__)

def gunzip(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)

def rmse(actual, pred):
    return sqrt(mean_squared_error(actual, pred))

def maximumZero(x):
    if x > 0:
        return x
    else:
        return 0

def baseline_rolling(df, year_roll):
    df_c_all = df[['lon','lat', 'year', 'yield']]
    df_c = df_c_all.groupby(['lon', 'lat']).size().reset_index()
    df_c.columns = ['lon', 'lat', 'count']
    df_c = df_c[df_c['count'] > 25]
    df_coords = df_c_all.merge(df_c, on = ['lon', 'lat'], how = 'inner')
    df_coords['rolling_yield3'] = df_coords['yield'].rolling(year_roll).mean()
    df_coords.loc[df_coords['year'] < (1981 + year_roll), 'rolling_yield3'] = df_coords['yield']
    df_coords['year'] = df_coords['year'] - 1
    df_coords = df_coords[['lon', 'lat', 'year', 'rolling_yield3']]
    df_out = df_c_all.merge(df_coords, on=['lon','lat', 'year'], how = 'inner')
    df_out['errors'] = abs(df_out['yield'] - df_out['rolling_yield3'])
    return df_out


def recenter_lon(lon):
    if lon > 180:
        return lon - 360
    else:
        return lon

def get_data(crop, type='mix'):
    global df_train, df_test
    datapath = "C:\\Users\\Andreas Langholz\\Yieldcaster\\Data\\Processed\\"
    croppath = datapath + crop + '\\'
    if type == 'mix':
        df_train = pd.read_csv(croppath + 'df_' + crop + '_historical_mix_train90.csv', index_col=0)
        df_test = pd.read_csv(croppath + 'df_' + crop + '_historical_mix_test10.csv', index_col=0)
    elif type == 'pix':
        df_train = pd.read_csv(croppath + 'df_' + crop + '_historical_pix_train90.csv', index_col=0)
        df_test = pd.read_csv(croppath + 'df_' + crop + '_historical_pix_test10.csv', index_col=0)
    elif type == '1year':
        df_train = pd.read_csv(croppath + 'df_' + crop + '_historical_1yearshold_pre.csv', index_col=0)
        df_test = pd.read_csv(croppath + 'df_' + crop + '_historical_1yearshold_post.csv', index_col=0)
    elif type == '5year':
        df_train = pd.read_csv(croppath + 'df_' + crop + '_historical_5yearshold_pre.csv', index_col=0)
        df_test = pd.read_csv(croppath + 'df_' + crop + '_historical_5yearshold_post.csv', index_col=0)
    else:
        print('Wrong type')

    return df_train, df_test


def df_mth_to_year(df):
    df_var_horizontal = df.sort_values(by=['lon', 'lat', 'year', 'month'])
    df_var_horizontal.drop('month', axis=1, inplace=True)

    try:
        df_var_horizontal.drop('time', axis=1, inplace=True)
    except:
        print('no time column')

    df_var_horizontal = (df_var_horizontal.set_index(['lon', 'lat', 'year',
                                                      df_var_horizontal.groupby(['lon', 'lat', 'year']).cumcount().add(
                                                          1)])
                         .unstack()
                         .sort_index(axis=1, level=1))
    df_var_horizontal.columns = [f'{a}{b}' for a, b in df_var_horizontal.columns]
    df_var_horizontal = df_var_horizontal.reset_index()
    return df_var_horizontal


def fast_join(df1, df2, list_to_join_on, how='outer'):
    df1.set_index(list_to_join_on, inplace=True)
    df2.set_index(list_to_join_on, inplace=True)
    df_out = df1.join(df2, how=how, on=list_to_join_on, lsuffix='left', rsuffix='right')
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df_out = df_out.reset_index()
    return df_out

# Not used
def sub_mask(df, mask='yield'):
    if mask == 'yield':
        df_mask = pd.read_csv("C:\\Users\\langh\\Individual_project\\Yieldcaster\\data\\interrim\\yield_mask.csv",
                              index_col=0)
    if mask == 'land':
        df_mask = pd.read_csv("C:\\Users\\langh\\Individual_project\\Yieldcaster\\data\\interrim\\landmask.csv",
                              index_col=0)
    df_out = df.merge(df_mask, on=['lon','lat'], how = 'inner')
    return df_out

def subset(df, month_out = True, year = 2000, month = 5):
    if month_out:
        df_out = df[(df['year'] == year) & (df['month'] == month)]
    else:
        df_out = df[df['year'] == year] 
    return df_out


def timeToDays(timestr):
    if type(timestr) == str:
        days_span = re.search("days", timestr)
        days_str = timestr[:days_span.span()[0] - 1]
        hour_str = timestr[days_span.span()[1] + 1:days_span.span()[1] + 3]
        return int(days_str) if hour_str == '' else int(days_str) + round(int(hour_str) / 24)
    else:
        try:
            return int(timestr)
        except:
            return timestr
