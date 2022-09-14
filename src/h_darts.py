""" 
MSc. Project 2022 - Andreas Langholz - Darts related functions and classes

Description: All functions used in the model related to the Darts framework including the timeseries API
OBS: The primary operations are done in the TimeseriesDarts class which carries the parameters 
for a specific dataset in order to recreate the run at a later stage
"""


from pytorch_lightning.callbacks import Callback
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from darts.timeseries import concatenate
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from src.h_utils import *
from sklearn.preprocessing import StandardScaler

def pred_to_df(prediction):
    """ Converts darts prediction timeseries to dataframe
    Loops over the timeseries and extracts static covariates and prediction results
    to identify the pixels and crops
    """

    df_pred = prediction[0].pd_dataframe().reset_index()
    df_static = prediction[0].static_covariates
    df_static = df_static.loc[df_static.index.repeat(len(df_pred))]
    df_static = df_static.reset_index(drop = True)
    df_results = pd.concat([df_static, df_pred], axis = 1)
    for i in range(1,len(prediction)):
        df_pred = prediction[i].pd_dataframe().reset_index()
        df_static = prediction[i].static_covariates
        df_static = df_static.loc[df_static.index.repeat(len(df_pred))]
        df_static = df_static.reset_index(drop = True)
        df_res = pd.concat([df_static, df_pred], axis = 1)
        df_results = pd.concat([df_results, df_res], axis = 0)
    return df_results

class LossLogger(Callback):
    """ Callback module to the PT lightning trainer
    logging the evaluation losses at each epoxh and saving as a list 
    """
    def __init__(self):
        self.train_loss = []
        self.val_MSE = []
        self.val_MAE = []
        self.val_MAPE = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_MSE.append(float(trainer.callback_metrics["val_loss"])) # the regular function is MSELoss -> needs to be separated for quantile losses
        self.val_MAE.append(float(trainer.callback_metrics["val_MeanAbsoluteError"]))
        self.val_MAPE.append(float(trainer.callback_metrics["val_MeanAbsolutePercentageError"]))

class timeseries_darts:
    """ Main class taking care of the conversion between dataset and the final timeseries API used for the model
     """
    def __init__(self, horizon_pred, CROPS):
        self.horizon_pred = horizon_pred
        self.time_idx="year"
        self.target="yield"
        self.group_ids=["pixel_id", "sim"] + CROPS # Assumes onehot encoded crops
        self.static_reals=["lon", "lat", "harvest"]

    def get_training_ts(self, df_train, val_hist = False, train_hist = False):

        """ 
        Converts input dataframe (df_train) into the timeseries API for the darts model 
        """

        try:
            df_train.drop(0, inplace=True) # shouldnt happen, but just defensive measure
        except:
            pass 
        
        non_time_reals = [self.target] + [self.time_idx] + self.group_ids + self.static_reals
        self.time_varying_reals = [f for f in df_train.columns if f not in non_time_reals]
        
        df_train['sim'] = df_train['sim'].apply(lambda x : 1 if x == 'Y' else 0)
        df_train['pixel_id'] = df_train['pixel_id'].apply(int)
        df_train = df_train.apply(downcast_types)

        df_train.sort_values(self.group_ids + [self.time_idx], inplace=True)
        df_train.reset_index(drop=True, inplace=True)

        # Apply feature scaling to the covariates
        # only fit scalers on training to avoid information leackages on future series    
        self.scaler_cov = StandardScaler()

        df_not_scaled = df_train[[self.target] + ['year'] + self.group_ids]
        df_cov = df_train[self.static_reals + self.time_varying_reals]

        scaled = self.scaler_cov.fit_transform(df_cov.to_numpy())
        df_scaled = pd.DataFrame(scaled, columns=df_cov.columns)
        df_train = pd.concat([df_not_scaled, df_scaled], axis=1)

        #Split into train and and validation - only validating on simulated data and mixing depending on type
        cutoff_train = df_train.year.max() - self.horizon_pred 
        
        df_val = df_train[lambda x: x.sim == 1]
        
        if train_hist:
            df_train = df_train[lambda x: x.sim == 0]
        
        ts_target_train = TimeSeries.from_group_dataframe(
            df_train[lambda x: x.year <= cutoff_train][self.group_ids + [ self.time_idx] + self.static_reals + [self.target]], 
            group_cols = self.group_ids, 
            time_col= self.time_idx, 
            static_cols=self.static_reals)

        ts_cov_train = TimeSeries.from_group_dataframe(
            df_train[lambda x: x.year <= cutoff_train].drop(self.target, axis = 1), 
            group_cols = self.group_ids, 
            time_col= self.time_idx,
            static_cols=self.static_reals)
        
        self.train_cols = df_train.columns
        self.ts_target_train_sample = ts_target_train[0]
        self.ts_cov_train_sample = ts_cov_train[0]

    # TODO: Subsetting the validation set to only the smallest future series can save some memory!
    # .. Its probably not a lot but everything counts when you get errors..
    
        ts_target_val = TimeSeries.from_group_dataframe(
            df_val[self.group_ids + [self.time_idx] + self.static_reals + [self.target]], 
            group_cols = self.group_ids, 
            time_col= self.time_idx, 
            static_cols=self.static_reals)

        ts_cov_val = TimeSeries.from_group_dataframe(
            df_val.drop(self.target, axis = 1), 
            group_cols = self.group_ids, 
            time_col= self.time_idx,
            static_cols=self.static_reals)

        return ts_target_train, ts_target_val, ts_cov_train, ts_cov_val

    def get_prediction_ts(self, data, climate_model, scenario):

        """ Makes darts timeseries objects for the targets and covariates used in prediction
            As the the model is extrapolating the target we need the starting timeseries
        """
        # TODO: Following the split in the training data, this should also be separated into two functions
        # for (1) dataloading and preparation and (2) conversion to Darts TS API

        cwd = os.getcwd()
        int_path = cwd + "/data/interrim/"
        process_path = cwd + "/data/processed/"
        
        # Get the target datasets and ensure consistent pixel numbering
        df_target = pd.read_csv(
            process_path + 'df_train_target.csv', index_col=0)

        df_target = df_target[df_target.crop.isin(data.crops)]
        yield_mask = pd.read_csv(int_path + 'yieldmask.csv', index_col=0)
        df_target = df_target.merge(yield_mask, on=['lon', 'lat'], how='inner')
        df_g = df_target.groupby(['lon', 'lat', 'crop']).size()
        df_g = df_g[df_g == df_g.max()]
        df_g = df_g.reset_index()
        df_target = df_target.merge(df_g, on=['lon', 'lat', 'crop'], how='inner')
        df_target.reset_index(drop=True, inplace=True)
        df_target.drop(0, axis=1,inplace=True)

        # Copy pixels and static variables to fill out full timespan (1982 - 2100)
        df_t = df_target.groupby(['lon', 'lat', 'pixel_id', 'harvest', 'crop']).size().reset_index()
        df_t = df_t.drop(0,axis=1)
        min_year = df_target.year.min()
        max_year = 2100

        df_out = df_t.copy()
        df_out['year'] = min_year

        for year in range(min_year + 1, max_year + 1):
            df_o = df_t.copy()
            df_o['year'] = year
            df_out = pd.concat([df_out, df_o], axis = 0)

        df_out = df_out.reset_index(drop=True)

        # Load historical simualted data and the relevant climate simulation
        df_features_hist = pd.read_csv(
            process_path + 'avg_historical_pred_' + data.type +'.csv', index_col=0)

        df_features_pred = pd.read_csv(
            process_path + climate_model + '_' + scenario + '_pred_' + data.type + '.csv', index_col=0)

        # Cut relevant pixels from the featuresets
        df_go = df_g.drop([0, 'crop'], axis=1)
        df_go = df_go.groupby(['lon', 'lat']).size().reset_index().drop(0,axis=1)

        df_features_h = df_features_hist.merge(df_go, on=['lon', 'lat'], how='inner')
        df_features_p = df_features_pred.merge(df_go, on=['lon', 'lat'], how='inner')
        df_features_all = pd.concat([df_features_h, df_features_p], axis = 0)

        # Merge all the datasets
        df_all = df_out.merge(df_features_all, on=['lon', 'lat', 'year'])
        df_yield = df_target[['yield', 'pixel_id', 'year', 'crop']]
        df_all = df_all.merge(df_yield, on=['pixel_id', 'year', 'crop'], how='outer')
        df_all['yield'] = df_all['yield'].fillna(0)
        df_all['sim'] = 1

        # Make columns into PCA
        if data.pca:
            df_all = data.convert_pca(df_all)

        # Subset only used columns for model training (From cutoff)
        df_pred = df_all[['crop'] + data.non_features + data.target + self.time_varying_reals]
        df_pred = data.encode_crop(df_pred)
        df_pred.drop('crop', axis=1, inplace=True)

        # Scale vars if not PCA already
        if not data.pca:
            df_pred.sort_values(self.group_ids + [self.time_idx], inplace=True)
            df_pred.reset_index(drop=True, inplace=True)
            df_not_scaled = df_pred[[self.target] + ['year'] + self.group_ids]
            df_cov = df_pred[self.static_reals + self.time_varying_reals]
            scaled = self.scaler_cov.transform(df_cov.to_numpy())
            df_scaled = pd.DataFrame(scaled, columns=df_cov.columns)
            df_pred = pd.concat([df_not_scaled, df_scaled], axis=1)

        # Make timeseries 
        cutoff = df_target.year.max()
        df_pred = df_pred.apply(downcast_types)

        ts_target = TimeSeries.from_group_dataframe(
            df_pred[lambda x: x.year <= cutoff][self.group_ids + [self.time_idx] + self.static_reals + [self.target]], 
            group_cols = self.group_ids, 
            time_col= self.time_idx, 
            static_cols=self.static_reals)

        ts_cov = TimeSeries.from_group_dataframe(
            df_pred.drop(self.target, axis = 1), 
            group_cols = self.group_ids, 
            time_col= self.time_idx,
            static_cols=self.static_reals)

        return ts_target, ts_cov, df_out, df_target

## The below are obsolete and now taken care of by the TimeseriesDarts anc DataMaker classes respectfully

def df_to_scaled_ts(df_model, target, conv_pca = False):
    ts_target, ts_cov =  convert_timeseries_onehot(df_model, target)
    # Fit the scalers and transforms seies
    scaler_target = Scaler()
    scaler_target.fit(ts_target)
    ts_target_sc = scaler_target.transform(ts_target)
    scaler_cov = Scaler()
    if not conv_pca:
        scaler_cov.fit(ts_cov)
        ts_cov_sc = scaler_cov.transform(ts_cov)
    else: 
        ts_cov_sc = ts_cov
    scalers = {"target": scaler_target, "cov": scaler_cov}
    return ts_target_sc, ts_cov_sc, scalers

def split_after_list(ts, cutoff):
    input = []
    prediction = []
    for i in range(0, len(ts)):
        inp, pred = ts[i].split_after(cutoff)
        input.append(inp)
        prediction.append(pred)
    return input, prediction

def concat_list_ts(ts_train, ts_test):
    out = []
    for i in range(0, len(ts_train)):
        out.append(concatenate([ts_train[i], ts_test[i]]))
    return out

# For Darts
def split_train_test_ts(ts_target_sc, ts_cov_sc, cutoff):
    ts_target_train_sc, _ = split_after_list(ts_target_sc, cutoff)
    ts_target_test_sc = ts_target_train_sc
    ts_cov_train_sc, ts_cov_test_sc = ts_cov_sc, ts_cov_sc
    return ts_target_train_sc, ts_target_test_sc, ts_cov_train_sc, ts_cov_test_sc

# Obsolete.. converting straight in datamaker class
def convert_timeseries(df_train, target = 'yield'):
    targets = []
    cov = []
    crops = df_train.crop.unique()
    for crop in crops:    
        df_c = df_train[df_train.crop == crop]
        df_c.drop('crop', axis = 1, inplace = True)
        for i in df_c.pixel_id.unique():
            obs = df_c[df_c.pixel_id == i]
            dst = TimeSeries.from_dataframe(obs, 'year', target)
            dsc = obs.drop(target, axis=1)
            dsc = TimeSeries.from_dataframe(dsc, 'year')
            targets.append(dst)
            cov.append(dsc)
    return targets, cov

def convert_timeseries_group(df_train, target = 'yield'):
    time = 'year'
    group_ids=["crop", "pixel_id", "sim"]
    static_categoricals=["crop", "pixel_id", "sim"]
    static_reals=["lon", "lat", "harvest"]
    
    df_t = df_train[group_ids + [time] + static_categoricals + static_reals + [target]]
    df_cov = df_train.drop(target, axis = 1)
    
    targets = TimeSeries.from_group_dataframe(df_t, 
        group_cols = group_ids, 
        time_col=time, static_cols=static_reals)
    
    cov = TimeSeries.from_group_dataframe(df_cov, 
        group_cols = group_ids, 
        time_col=time, static_cols=static_reals)
    return targets, cov


def convert_timeseries_onehot(df_train, target):
    targets = []
    cov = []
    crops = df_train.crop.unique().tolist()
    drop = ['lon', 'lat', 'crop', 'harvest'] + target
    drop = drop + crops if len(crops) > 1 else drop
    for crop in crops:    
        df_c = df_train[df_train.crop == crop]
        for i in df_c.pixel_id.unique():
            obs = df_c[df_c.pixel_id == i]
            dst = TimeSeries.from_dataframe(obs, 'year', target)
            if len(crops) > 1:
                s_cov = ['lon', 'lat', 'harvest'] + crops
            else:
                s_cov = ['lon', 'lat', 'harvest']
            static_covs = obs[s_cov]
            dst = dst.with_static_covariates(static_covs.iloc[0])
            dsc = obs.drop(drop, axis=1)
            dsc = TimeSeries.from_dataframe(dsc, 'year')
            targets.append(dst)
            cov.append(dsc)
    return targets, cov

