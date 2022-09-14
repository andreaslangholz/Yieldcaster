"""
MSc. Project 2022 - Andreas Langholz - Training script for the TFT Reference architecture model
Description: Trains the model on the inputtet hyperparameters and saves as a dctionary together with
the accompanying timeseries_darts and data_maker classes needed for runnign the predictions
"""

from multiprocessing import freeze_support
from darts.utils.likelihood_models import QuantileRegression
from darts.models import TFTModel
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

def main():
    # Name of model
    model_name = 'wheat_historical_shorts'

    # Script control parameters
    # Set prediction target 'yield, 'product' or both
    TARGET = ['yield']
    CROPS = ['wheat_winter', 'wheat_spring'] #, 'rice', 'maize', 'soy']    # set to target crops
    PCT_DEBUG = 1     # Pct of pixels used
    AREA = 'all'      # ['all', 'france', 'europe', 'china', 'US']
    TRAIN_HIST = False

    # drop variables with low correlation
    CORR_THR = 0.05    # threshold for minimum correlation between feature and target

    # Conversion to PCA
    CONV_PCA = False             # Use PCA faetures instead of regular
    N_PCA = 30                  # Number of PCAs

    # Timeseries parameters (total timeseris years <= input_ch + hor_train + hor_test)
    INP_LENGTH = 4     # Length of timeseries training period
    HOR_TRAIN = 2       # Length of forecasting period for training
    TYPE = 'cruts'
    DETREND = False

    # Model hyperparameters
    EPOCHS = 30
    HID_SIZE = 256
    BATCH_S = 16
    LEARN_RATE = 0.001
    LSTM_L = 3
    MAX_S = 1
    N_HEADS = 4
    DROPOUT = 0.2

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
    model = TFTModel(
        input_chunk_length=INP_LENGTH,
        output_chunk_length=HOR_TRAIN,
        hidden_size=HID_SIZE,
        lstm_layers=LSTM_L,
        num_attention_heads=N_HEADS,
        dropout=DROPOUT,
        batch_size=BATCH_S,
        n_epochs=EPOCHS,
        likelihood=likelihood,
        torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError(), MeanSquaredError()]),
        loss_fn=loss_fn,
        random_state=42,
        pl_trainer_kwargs={"callbacks": [my_stopper, loss_logger]},
    )

    model.fit(ts_target_train,
        future_covariates=ts_cov_train, 
        val_series=ts_target_val,
        val_future_covariates=ts_cov_val, 
        verbose=True, 
        max_samples_per_ts=4, 
        num_loader_workers=4,
    )

    model.train_sample[4]

    # Save model, loss and specs for analysis and prediction
    # Save the best model according to validation set
    path = cwd + '/models/' + model_name
    model.save(path + '.pth.tar')
    model.epochs_trained
    model_dict = { 'data_maker': data,
                    'tsd': tsd}
    pickl_save(model_dict, path + '_dict')
    df_loss = pd.DataFrame({'epoch': [i+1 for i in range(len(loss_logger.val_MSE))], 
                                'train_MSE_loss': [loss_logger.val_MSE[0]] + loss_logger.train_loss,
                                'val_MSE_loss': loss_logger.val_MSE,
                                'val_MAE_loss': loss_logger.val_MAE,
                                'val_MAPE_loss': loss_logger.val_MAPE})
                            
    df_loss.to_csv(cwd +
                "/data/outputs/hist_errors/" + model_name + "_loss.csv")


if __name__ == "__main__":
    freeze_support()
    print('Running as main w Freeze support')
    main()
