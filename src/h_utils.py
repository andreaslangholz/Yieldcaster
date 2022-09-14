
"""  
MSc. Project 2022 - Andreas Langholz - General utilities
Description:
The utilities are split in:
(1) Functions used in datacleaning
(2) Classes and functions related to producing final dataset 
ready for training/prediction. The data_maker class most importantly carries the parameters
needed for replicating datasets when saving the model
(3) Other general utility functions
"""

import pandas as pd
import re
import math
import gzip
import numpy as np
from datetime import datetime as dt
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import os
from src.h_geo import geo_subset
import gc

""" 
(1) Data cleaning functions
"""

def gunzip(source_filepath, dest_filepath, block_size=65536):
    # Function to open zip archives sequentially for large files
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)

def max_zero(x):
    # Returns max(x,0) to be applied in row tranformation
    if x > 0:
        return x
    else:
        return 0

def recenter_lon(lon):
    """ Recenters the longitude to ensure that all coordinates span 
    the same field (lon: [-180;180], lat: [-90;90])
     """
    if lon > 180:
        return lon - 360
    else:
        return lon

def convert_to_dt(x):
    """ Converts string to date-time wuth standard format
     """
    return dt.strptime(str(x), '%Y-%m-%d %H:%M:%S')

def df_mth_to_year(df):
    """ Converts paneldata dataframe with rows per month into df with rows per pixel per year
    must contain columns ['lon', 'lat', 'year', 'month'] used for identifying pixels and month/years
    final dataframe will have ['lon', 'lat', 'year'] + features with index var1,..,var12 for each month
    """

    df_var_horizontal = df.sort_values(by=['lon', 'lat', 'year', 'month'])
    df_var_horizontal.drop('month', axis=1, inplace=True)

    try: #protective, but not neccessary
        df_var_horizontal.drop('time', axis=1, inplace=True)
    except:
        print('no time column')

    df_var_horizontal = (df_var_horizontal.set_index(['lon', 'lat', 'year',
                                                      df_var_horizontal.groupby(['lon', 'lat', 'year']).cumcount().add(
                                                          1)])
                         .unstack()
                         .sort_index(axis=1, level=1))
    df_var_horizontal.columns = [
        f'{a}{b}' for a, b in df_var_horizontal.columns]
    df_var_horizontal = df_var_horizontal.reset_index()
    return df_var_horizontal


def fast_join(df1, df2, list_to_join_on, how='outer'):
    """ Joins two dataframes on the list_to_join columns by converting 
    to indices. Generally ~4x faster than merge
    """
    
    df1.set_index(list_to_join_on, inplace=True)
    df2.set_index(list_to_join_on, inplace=True)
    df_out = df1.join(df2, how=how, on=list_to_join_on,
                      lsuffix='left', rsuffix='right')
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df_out = df_out.reset_index()
    return df_out

def round_to_quarters(x):
    """ 
    Rounds the x value to the nearest .25 or .75 value 
    Used in the 
    """
    x_abs = abs(x)
    dif = x_abs - np.round(x_abs)
    new_dec = 0.75 if dif < 0 else 0.25
    x_out = math.copysign(1, x) * (math.floor(x_abs) + new_dec)
    return x_out

def subset(df, month_out=True, year=2010, month=5):
    if month_out:
        df_out = df[(df['year'] == year) & (df['month'] == month)]
    else:
        df_out = df[df['year'] == year]
    return df_out

def time_to_days(timestr):
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

""" 
(2) Classes and functions related to producing final dataset 
"""
class data_maker:
    def __init__(self, target, crops, detrend=False, pct_debug=1, area='all', corr_thr=0, n_pca=30, type = 'cruts', encode=False, all_years=False):
        self.target = target
        self.detrend = detrend # not used
        self.crops = crops
        self.pct_debug = pct_debug
        self.area = area
        self.corr_thr = corr_thr
        self.pca_transformer = PCA(n_components=n_pca)
        self.pca_scaler = MinMaxScaler()
        self.pca = False
        self.type = type
        self.encode = encode
        self.all_years = all_years
        self.df_train = None
        self.df_pred = None
        self.int_path = os.getcwd() + "/data/interrim/"
        self.process_path = os.getcwd() + "/data/processed/"
        self.non_features = ['lon', 'lat', 
                             'year', 'harvest', 'pixel_id', 'sim']
        self.features = None
    
        """ Data class encapsulating the parameters used for generating the model training and prediction datasets
        
        Parameters
        ----------
        target
            The target variable that the model is trained to predict (currently only using 'yields')         
        crops
            The crops to include in the dataset        
        detrend
            Whether to use detrended datasets (not currently implemented)
        pct_debug
            pct of pixels randomly chosen included in the final dataset
        area
            subset on a geographical area
        corr_thr
            minimum correlation value (pearson) between an individual feature and the target
            in order for the feature to be included in the dataset
        n_pca
            number of pca features used for decomposition (if pca is used)
        type 
            type of dataset used for features ('cruts' or 'etccdi')
        encode
            if crops should be onhot encoded
        all_years
            only include timeseries with full length (1982 - 2015)
        """ 
        
    def corr_correct(self, df_all):
        features = []
        df_corr = df_all.drop(self.non_features, axis=1)
        for crop in self.crops:
            df_corr_crop = df_corr[df_corr.crop == crop]
            df_corr_crop = df_corr_crop.drop('crop', axis=1)
            for var in df_corr_crop.columns:
                for t in self.target:
                    if df_corr_crop[t].corr(df_corr_crop[var]) > self.corr_thr:
                        if var not in features:
                            features.append(var)
        features = [x for x in features if (x not in self.target)]
        print('features after cutting off at ', self.corr_thr, len(features))
        out = self.target + self.non_features + ['crop']
        df_out = df_all[out + features]
        self.features = features
        return df_out

    def encode_crop(self, df):
        df = df.reset_index(drop=True)
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(df[['crop']])
        encoder_df = pd.DataFrame(encoder.transform(df[['crop']]).toarray())
        encoder_df.columns = encoder.get_feature_names()
        encoder_df.columns = encoder_df.columns.str.strip('x0_')
        final_df = df.join(encoder_df)
        return final_df

    def convert_pca(self, df_all, fit_pca=False):
        out = self.non_features + self.target + ['crop']
        df_feat = df_all.drop(out, axis=1).copy()
        if fit_pca:
            df_feat = self.pca_scaler.fit_transform(df_feat)
            res_pca = self.pca_transformer.fit_transform(df_feat)
        else:
            df_feat = self.pca_scaler.transform(df_feat)
            res_pca = self.pca_transformer.transform(df_feat)
        df_pca = pd.DataFrame(res_pca)
        df_pca = df_pca.add_prefix("pca")
        df_out = df_all[out]
        df_out = pd.concat([df_out, df_pca], axis=1)
        return df_out

    def subset_pct_df(self, df, pct_debug):
        pixels = df.groupby(
            ['lon', 'lat'], as_index=False).size().drop('size', axis=1)
        num_rows = len(pixels)

        num_train = round(num_rows * pct_debug)
        idx = np.arange(num_rows)

        np.random.shuffle(idx)
        train_idx = idx[0:num_train]

        df_coords_train = pixels.iloc[train_idx]
        df_model = df.merge(df_coords_train, on=['lon', 'lat'], how='inner')
        return df_model

    def get_training_data(self, pca=False):
        """ Gets the combined training df based on the specified parameters in the
        initialisation of the class
        """

        df_target = pd.read_csv(
            self.process_path + 'df_train_target.csv', index_col=0)
            
        df_target = df_target[df_target.crop.isin(self.crops)]

        # Not neccessary to subset on yieldmask again
        # yield_mask = pd.read_csv(self.int_path + 'yieldmask.csv', index_col=0)
        # yield_mask['pixel_id'] = yield_mask['pixel_id'].apply(
        #     str)  # PtF is only taking categoricals as groups
        # df_target = df_target.merge(yield_mask, on=['lon', 'lat'], how='inner')
        # del yield_mask

        # Only returns dataset with all years present
        if self.all_years == True:
            df_g = df_target.groupby(['lon', 'lat', 'crop']).size()
            df_g = df_g[df_g == df_g.max()]
            df_g = df_g.reset_index()
            df_target = df_target.merge(df_g, on=['lon', 'lat', 'crop'], how='inner')
            df_target.reset_index(drop=True, inplace=True)

        # Pick the type of featureset to use
        if self.type == 'cruts':

            df_features = pd.read_csv(
                self.process_path + 'df_train_features_cruts' + '.csv', index_col=0)
            df_all = df_target.merge(df_features, on=['lon', 'lat', 'year'])
            df_all['sim'] = 'N'

            df_features_sim = pd.read_csv(
                self.process_path + 'avg_historical_pred_cruts' + '.csv', index_col=0)
            
            df_all_sim = df_target.merge(
                df_features_sim, on=['lon', 'lat', 'year'], how='inner')
            
            df_all_sim['sim'] = 'Y'            
            df_all = pd.concat([df_all, df_all_sim], axis=0)
            df_all = df_all.sample(frac=1).reset_index(drop = True) # Shuffle the rows
            
            del df_features, df_all_sim, df_features_sim
            gc.collect()
        
        elif self.type == 'etccdi':
            df_features_sim = pd.read_csv(
                self.process_path + 'avg_historical_pred_etccdi.csv', index_col=0)
            
            df_all = df_target.merge(
                df_features_sim, on=['lon', 'lat', 'year'], how='inner')
            
            df_all['sim'] = 'Y'
            del df_features_sim
            gc.collect()
        
        else:
            print('no such data')

        # Include simulated historical averages in the training set
        df_all = self.convert_df(df_all, pca)

        self.last_train_year = df_all.year.max()
        self.train_sample = df_all.loc[0]
        
        return df_all

    def convert_df(self, df_all, pca):
        # Subset on geography
        df_all = geo_subset(df_all, self.area)

        # cut pct of pixels
        if self.pct_debug < 1: 
            df_all = self.subset_pct_df(df_all, self.pct_debug)

        # Convert to pca
        if pca:
            df_all = self.convert_pca(df_all, fit_pca=True)
            self.pca = True
        
        # Cut features with low correlation
        if self.corr_thr > 0:
            df_all = self.corr_correct(df_all)

        # Onehot encode crops
        if self.encode:
            df_all = self.encode_crop(df_all)
            try:
                df_all.drop('crop', inplace = True, axis=1)
            except:
                pass
        
        return df_all

""" 
(3) Other general utility functions
"""

def pickl_load(path):
    """ loads a pickl object """
    with open(path + '.pickle', 'rb') as f:
        return pickle.load(f)

def pickl_save(obj, path):
    """ saves a pickl object """
    with open(path + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def downcast_types( s ):
    """ Downcasts a column to smallest float or integer if possible
    To be used in df.transform() to downcast full dataframe 
     """
    if s.dtype == 'float64':
        return pd.to_numeric( s, downcast='float')
    elif s.dtype == 'int64':
        return pd.to_numeric( s, downcast='integer')
    else:
        return s

def convert_results(df_results):
    """ Calculates production and values from predicted yields
    """

    prices = {
        'wheat_spring': 215.2,
        'wheat_winter': 215.2,
        'soy': 405.9,
        'rice': 268.4,
        'maize': 176.0
    }

    df_results['production'] = df_results['yield'] * df_results['harvest']

    df_results['value'] = 0

    for crop in df_results.crop.unique():
        valuef = lambda x : x.production * prices[crop]
        df_results['value'] = df_results.apply(valuef, axis=1)

    return df_results

