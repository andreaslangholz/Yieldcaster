
""" 
MSc. Project 2022 - Andreas Langholz - Model comparison script
Description: Loops over different experiments in the model comparsion and outputs related MSE, MAE and MAPE values per epoch
""" 



from multiprocessing import freeze_support
from darts.utils.likelihood_models import QuantileRegression
from darts.models import TFTModel
import pandas as pd
import os
import sys
import torch.nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
from darts import TimeSeries
cwd = os.getcwd()
sys.path.insert(0, cwd)
from src.h_utils import *
from src.h_darts import *
from src.tft_tcn import *
import gc

def main():
    # Script control parameters
    # Set prediction target 'yield, 'product' or both
    TARGET = ['yield']
    PCT_DEBUG = 0.3     # Pct of dataset used
    AREA = 'all'      # ['all', 'france', 'europe', 'china', 'US']
    # Conversion to PCA
    N_PCA = 30                  # Number of PCAs
    # Timeseries parameters (total timeseris years <= input_ch + hor_train + hor_test)
    INP_LENGTH = 7     # Length of timeseries training period
    HOR_TRAIN = 3       # Length of forecasting period for training
    # Model hyperparameters
    EPOCHS = 15
    HID_SIZE = 512  
    BATCH_S = 16
    LSTM_L = 3
    N_HEADS = 4
    DROPOUT = 0.2
    THR = 0
    # Set parameter to use quantile regression (probabilistic) or deterministic loss
    likelihood = None
    loss_fn = torch.nn.MSELoss()
    detrend = False
    # Test search
    thresholds_test = [0.05] #, 0.1, 0.2]
    pca_test = [False] #, False]
    crops_test = ['wheat_winter', 'rice'], ['wheat_winter'], ['rice'] #, ['rice'], ['maize'], ['soy']]
    type_test = ['cruts_hist', 'cruts_mix'] #, 'etccdi']
    model_custom = [True, False]
    loss_metrics = []

#   for pca_t in pca_test:
    for model_c in model_custom:
        for crops in crops_test:
            for data_type in type_test:
                # Data constructor and manipulation classes
                data_t = data_type if data_type == 'etccdi' else 'cruts'
                data = data_maker(TARGET, crops, detrend, PCT_DEBUG, AREA, THR, N_PCA, type= data_t, all_years=True, encode=True)
                df_train = data.get_training_data(pca=False)
                print('running: ', crops, data_type, 'Thresholds: ', THR)
                if data_type == 'cruts_hist':
                    hist_data_test = True
                    val_hist = True
                else:
                    hist_data_test = False
                    val_hist = False
                
                gc.collect()
                tsd = timeseries_darts(HOR_TRAIN, crops)
                ts_target_train, ts_target_val, ts_cov_train, ts_cov_val = tsd.get_training_ts(df_train,val_hist, hist_data_test)
                del df_train
                gc.collect()
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
                # Make model
                if model_c:
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
                        random_state=42,
                        pl_trainer_kwargs={"callbacks": [my_stopper, loss_logger]},
                    )
                else:    
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
                        loss_fn=loss_fn,
                        random_state=42,
                        torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()]),
                        pl_trainer_kwargs={"callbacks": [my_stopper, loss_logger]},
                    )
                print('Fitting model')
                model.fit(ts_target_train,
                    future_covariates=ts_cov_train, 
                    val_series=ts_target_val,
                    val_future_covariates=ts_cov_val, 
                    verbose=True, 
                    max_samples_per_ts=1, 
                    num_loader_workers=4,
                )

                print('Logging losses')
                df_loss = pd.DataFrame({'epoch': [i+1 for i in range(len(loss_logger.val_MSE))], 
                    'train_MSE_loss': [loss_logger.val_MSE[0]] + loss_logger.train_loss,
                    'val_MSE_loss': loss_logger.val_MSE,
                    'val_MAE_loss': loss_logger.val_MAE,
                    'val_MAPE_loss': loss_logger.val_MAPE})

                m = 'custom' if model_c else 'reference'
                metrics = dict(model = m, crops=crops, data_type = data_type, values =df_loss )
                loss_metrics.append(metrics)
                print('Losses for model: ', df_loss)
                path = cwd + '/models/parameter_tests'
                pickl_save(loss_metrics, path)
                del model, ts_target_train, ts_target_val, ts_cov_train, ts_cov_val
                gc.collect()


if __name__ == "__main__":
    freeze_support()
    print('Running as main w Freeze support')
    main()
