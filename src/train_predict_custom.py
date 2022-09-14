"""
MSc. Project 2022 - Andreas Langholz - Train and forecast on the custom TFT-TCN model
Description: Trains the model on the inputtet hyperparameters loops over the scenarios for predictions
outputs the model training losses and the crop predictions per scenario
"""


from multiprocessing import freeze_support
from darts.utils.likelihood_models import QuantileRegression
import pandas as pd
import os
import sys
from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
import torch.nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
cwd = os.getcwd()
sys.path.insert(0, cwd)
from src.h_utils import *
from src.h_darts import *
from src.tft_tcn import *

# def main():
    
# Name of model
model_name = 'custom_pred_maize_soy'
# Script control parameters
# Set prediction target 'yield, 'product' or both
TARGET = ['yield']
CROPS = ['maize', 'soy']#['wheat_winter', 'wheat_spring', 'rice']# , 'maize', 'soy']    # set to target crops
PCT_DEBUG = 1     # Pct of dataset used
AREA = 'all'      # ['all', 'france', 'europe', 'china', 'US']
VAL_HIST = False
TRAIN_HIST = False

# drop variables with low correlation
CORR_THR = 0    # threshold for minimum correlation between feature and target
# Conversion to PCA
CONV_PCA = False             # Use PCA faetures instead of regular
N_PCA = 30                  # Number of PCAs
# Timeseries parameters (total timeseris years <= input_ch + hor_train + hor_test)
INP_LENGTH = 10     # Length of timeseries training period
HOR_TRAIN = 4       # Length of forecasting period for training
TYPE = 'cruts'
DETREND = False

# Model hyperparameters
EPOCHS = 30
HID_SIZE = 256
BATCH_S = 16
LSTM_L = 3
DROPOUT = 0.2

# For prediction
CLIMATE_MODEL = 'avg'
SCENARIOS = ['ssp126', 'ssp245', 'ssp370','ssp585']

# Set parameter to use quantile regression (probabilistic) or deterministic loss
PROBABILISTIC = False   # if false, model uses MSELoss as loss_fn
QUANTILES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]


# Data constructor and manipulation classes
print('Loading dataset')
data = data_maker(TARGET, CROPS, DETREND ,PCT_DEBUG, AREA, CORR_THR, N_PCA, type = TYPE, all_years=True, encode=True)
df_train = data.get_training_data(CONV_PCA)
# Reference arch only supports reals so we have to encode them

print('Converting to timeseries')
tsd = timeseries_darts(HOR_TRAIN, CROPS)
ts_target_train, ts_target_val, ts_cov_train, ts_cov_val = tsd.get_training_ts(df_train, val_hist=VAL_HIST,train_hist=TRAIN_HIST)


print('Training model!')
loss_logger = LossLogger()
# stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
# a period of 5 epochs (`patience`)
my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=3,
    min_delta=0.05,
    mode='min',
)

# Set loss function
if PROBABILISTIC:
    likelihood = QuantileRegression(
        quantiles=QUANTILES
    )
    loss_fn = None
else:
    likelihood = None
    loss_fn = torch.nn.MSELoss()

# Make model
print("New model: " + model_name)
model = TFTModelCustom(
    input_chunk_length=INP_LENGTH,
    output_chunk_length=HOR_TRAIN,
    hidden_size=HID_SIZE,
    lstm_layers=LSTM_L, # Not used in custom but keep in to not mess with parameterisation flow
    num_attention_heads=4,
    dropout=DROPOUT,
    batch_size=BATCH_S,
    n_epochs=EPOCHS,
    add_relative_index=False,
    add_encoders=None,
    torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError(), MeanSquaredError()]),
    loss_fn=loss_fn,
    likelihood=likelihood,
    random_state=42,
    pl_trainer_kwargs={"callbacks": [my_stopper, loss_logger]},
)

model.fit(ts_target_train,
    future_covariates=ts_cov_train, 
    val_series=ts_target_val,
    val_future_covariates=ts_cov_val, 
    verbose=True, 
    max_samples_per_ts=1, 
    num_loader_workers=4,
)

tsd.time_varying_reals.remove(0)

ts_target, ts_cov, df_out, df_target = tsd.get_prediction_ts(data, CLIMATE_MODEL, scenario)

for scenario in SCENARIOS:
    ts_target, ts_cov, df_out, df_target = tsd.get_prediction_ts(data, CLIMATE_MODEL, scenario)
    # Passed the cov test -> Needs to retrain and save/load straight from Darts
    prediction = model.predict(n=80, 
        series= ts_target,
        future_covariates = ts_cov)
    df_pred_post = pred_to_df(prediction, data.target)
    df_pred_post['crop'] = df_pred_post[data.crops].idxmax(1)
    df_pred_post.drop(data.crops, inplace = True, axis = 1)
    df_pred_post.reset_index(drop=True)
    df_pred_post.drop(['lon', 'lat', 'harvest', 'sim'], inplace = True, axis = 1)
    df_pred_post = df_pred_post.merge(df_out, on=['pixel_id', 'year', 'crop'], how = 'inner')
    df_yield = pd.concat([df_pred_post, df_target], axis=0)
    df_yield = convert_results(df_yield)
    metrics = dict(scenario=scenario, df_pred = df_yield)
    dict_pred.append(metrics)
    path = cwd + '/models/' + model_name + '_pred'
    pickl_save(dict_pred, path)


if _name__ == "__main__":
    freeze_support()
    print('Running as main w Freeze support')
    main()