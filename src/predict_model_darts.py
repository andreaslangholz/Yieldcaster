
""" 
MSc. Project 2022 - Andreas Langholz - Darts preidction script
Description: Loads a dictionary (from train_model_darts.py) with data_amker and tim pre-trained darts reference model and predicts on the chosen climate scenario
""" 


import os
import sys
import pandas as pd
cwd = os.getcwd()
sys.path.insert(0, cwd)
from src.h_utils import *
from src.h_darts import *
from darts.models import TFTModel

int_path = cwd + "/data/interrim/"
process_path = cwd + "/data/processed/"

SCENARIO = 'ssp126'
CLIMATE_MODEL = 'avg'

model_name = "wheat_historical_shorts"

path = cwd + '/models/' + model_name

model = TFTModel.load(path + '.pth.tar')

model_dict = pickl_load(path + '_dict')
tsd = model_dict.get('tsd')
data = model_dict.get('data_maker')

ts_target, ts_cov, df_out, df_target = tsd.get_prediction_data(data, CLIMATE_MODEL, SCENARIO)

prediction = model.predict(n=50, 
    series= ts_target,
    future_covariates = ts_cov,
    num_samples = 5)

df_pred_post = pred_to_df(prediction, data.target)

df_pred_post['crop'] = df_pred_post[data.crops].idxmax(1)
df_pred_post.drop(data.crops, inplace = True, axis = 1)
df_pred_post.reset_index(drop=True)
df_pred_post.drop(['lon', 'lat', 'harvest', 'sim'], inplace = True, axis = 1)

df_pred_post = df_pred_post.merge(df_out, on=['pixel_id', 'year', 'crop'], how = 'inner')
df_yield = pd.concat([df_pred_post, df_target], axis=0)

df_yield.to_csv(cwd + "/data/outputs/" + model_name + SCENARIO + '.csv')

