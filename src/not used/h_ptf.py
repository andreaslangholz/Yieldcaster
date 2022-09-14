from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from src.h_geo import geo_subset
from src.h_utils import *

class timeSeriesPtf:
    def __init__(self, max_prediction_length, max_encoder_length, cutoff):
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.cutoff = cutoff
        self.time_idx="year"
        self.target="yield"
        self.group_ids=["crop", "pixel_id", "sim"]
        self.static_categoricals=["crop", "pixel_id", "sim"]
        self.static_reals=["lon", "lat", "harvest"]
        self.groups=["crop", "pixel_id", "sim"]
            
    def get_training_ts(self, df_model):
        columns = df_model.columns
        reals = [col for col in columns if col not in ['lon', 'lat', 'yield','crop', 'year', 'harvest','sim', 'pixel_id']]
        self.reals = reals
        self.parameters = dict(
            time_idx=self.time_idx,
            target=self.target,
            group_ids=self.group_ids,
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.max_prediction_length, # Always project MAX_PRD_LEN into validation set
            max_prediction_length=self.max_prediction_length,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            time_varying_known_reals=self.reals,
            # TODO: Test other normalisers / noscaling
            # -- There is an error reported on the variance of the normalised sequences        
            target_normalizer=GroupNormalizer( 
                groups=self.group_ids, 
                transformation="softplus"
            ), 
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        df_train = df_model[lambda x: x.year <= self.cutoff]

        training = TimeSeriesDataSet.from_parameters(self.parameters, df_train, predict = True, stop_randomization = False)        

        # Only validate on future simulated datasets
        df_val = df_model[df_model.sim == 'Y']
        validation = TimeSeriesDataSet.from_dataset(training, df_val, predict=True, stop_randomization = True)
    
        return training, validation
    
    def get_train_dataloaders(self, ts_train, ts_val, batch_s = 32):
        
        # Dataloaders
        train_load = ts_train.to_dataloader(train=True, batch_size=batch_s, num_workers=4)
        val_load = ts_val.to_dataloader(train=False, batch_size=batch_s, num_workers=4)

        return train_load, val_load

class modelPredictorPtf:
    def __init__(self, model, timeseries, crops, climate_model, scenario):
        int_path = 'c:\\Users\\Andreas Langholz\\Yieldcaster' + "/data/interrim/"
        process_path = 'c:\\Users\\Andreas Langholz\\Yieldcaster' + "/data/processed/"

        df_target = pd.read_csv(
            process_path + 'df_train_target.csv', index_col=0)
        df_target = df_target[df_target.crop.isin(crops)]
        
        yield_mask = pd.read_csv(int_path + 'yieldmask.csv', index_col=0)
        yield_mask['pixel_id'] = yield_mask['pixel_id'].apply(
                    str)  # PtF is only taking categoricals as groups
        
        df_static = df_target.drop(['year', 'yield'], axis=1)
        df_static = df_static.groupby(['lon', 'lat', 'harvest', 'crop']).count().reset_index()
        df_static = df_static.merge(yield_mask, on = ['lon', 'lat'])
        
        df_hist = pd.read_csv(
            process_path + climate_model + '_historical_pred.csv', index_col=0)
        df_hist = df_hist.apply(downcast_types)
        df_hist['sim'] = 'Y'
        df_hist['sim'] = pd.Categorical(df_hist['sim'])
        
        df_features_pred = pd.read_csv(
            process_path + climate_model + '_' + scenario + '_pred.csv', index_col=0)
        df_features_pred = df_features_pred.apply(downcast_types)
        df_features_pred['sim'] = 'Y'
        df_features_pred['sim'] = pd.Categorical(df_features_pred['sim'])

        self.df_static = df_static
        self.df_results = df_target.merge(yield_mask, on=['lon', 'lat'], how='inner')
        self.df_hist = df_hist
        self.df_pred = df_features_pred
        self.model = model
        self.crops = crops
        self.climate_model = climate_model
        self.scenario = scenario
        self.ts = timeseries

    def get_next_df(self):
        last_year = self.df_results.year.max()
        pred_year = last_year + self.ts.max_prediction_length
        encoder_year = last_year - self.ts.max_encoder_length
        df_r = self.df_results[lambda x : x.year >= encoder_year]
        for i in range(last_year + 1, pred_year + 1):
            df_s = self.df_static.copy()
            df_s['year'] = i
            df_s['yield'] = 0
            df_r = pd.concat([df_r, df_s])

        if encoder_year <= self.df_hist.year.max():
            df_hist_out = self.df_hist[lambda x : x.year >= encoder_year]
            df_hist_out = df_hist_out.merge(
                self.df_static, on=['lon', 'lat'], how='right')

            df_pred_out = self.df_pred[lambda x : x.year <= pred_year]
            df_pred_out = df_pred_out.merge(
                self.df_static, on=['lon', 'lat'], how='right')
            df_all_out = pd.concat([df_hist_out, df_pred_out])
        else:
            df_pred_out = self.df_pred[lambda x : x.year <= pred_year]
            df_pred_out = df_pred_out[lambda x : x.year >= encoder_year]
            df_all_out = df_pred_out.merge(
                self.df_static, on=['lon', 'lat'], how='right')

        df_out = df_r.merge(df_all_out, on=['pixel_id', 'year', 'crop', 'harvest', 'lon', 'lat'], how='inner')
        return df_out
    
    def predict_period(self):
        df_next = self.get_next_df()
        ts = TimeSeriesDataSet.from_parameters(self.ts.parameters, df_next, predict = True)        
        npredictions, index = self.model.predict(ts, mode="prediction", return_index = True, show_progress_bar = True)
        year = index.year.max()
        pred_all = pd.DataFrame(npredictions)[0]
        pred_all = pd.concat([index, pred_all], axis=1)
        pred_all.rename(columns={0:'yield'}, inplace=True)

        for i in range(1,self.ts.max_prediction_length):
            pred_year = pd.DataFrame(npredictions)[i]
            pred_year = pd.concat([index, pred_year], axis=1)
            pred_year['year'] = year + i
            pred_year.rename(columns={i:'yield'}, inplace=True)
            pred_all = pd.concat([pred_all, pred_year], axis = 0)

        pred_all.drop('sim', axis=1, inplace=True)
        df_static = self.df_static
        pred_all = pred_all.merge(df_static, on=['pixel_id', 'crop'])

        self.df_results = pd.concat([self.df_results, pred_all], axis=0)
        return None
    
    def predict_untill(self, max_year):
        current_year = self.df_results.year.max()
        while current_year < max_year:
            print('Predicting years {} to {}'.format(current_year +1, current_year+self.ts.max_prediction_length))
            self.predict_period()
            current_year = self.df_results.year.max()
        
        return self.df_results


    
    
