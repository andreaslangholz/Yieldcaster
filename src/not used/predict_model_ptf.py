import os
import sys
import pandas as pd
cwd = os.getcwd()
sys.path.insert(0, cwd)
from src.h_utils import *
from src.OLD.h_ptf import *

SCENARIO = 'ssp585'
CLIMATE_MODEL = 'avg'
PRED_YEAR = '2050'
CROPS = ['wheat_winter', 'wheat_spring', 'soy', 'maize', 'rice']

model_name = 'test_prediction_1year'
path = cwd + '/models/' + model_name

model_dict = pickl_load(path)
model = model_dict.get('model')
tsp = model_dict.get('tsp')

scenario = 'ssp245'
mp = modelPredictorPtf(model, tsp, CROPS, CLIMATE_MODEL, scenario)
df_results = mp.predict_untill(2030)
df_results = convert_results(df_results)
df_results['scenario'] = scenario

df_results.to_csv(cwd + "/data/outputs/" + model_name +'ssp245.csv')

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

df_out = pd.DataFrame()
for scenario in scenarios:
    mp = modelPredictorPtf(model, tsp, CROPS, CLIMATE_MODEL, scenario)
    df_results = mp.predict_untill(2050)
    df_results = convert_results(df_results)
    df_results['scenario'] = scenario
    df_out = pd.concat([df_out, df_results], axis = 0)

df_out.to_csv(cwd + "/data/outputs/" + model_name +'.csv')









