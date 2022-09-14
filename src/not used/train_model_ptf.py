print('start')
from pytorch_lightning.callbacks import ModelSummary
import os
import sys
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import QuantileLoss
cwd = os.getcwd()
sys.path.insert(0, cwd)
from src.OLD.h_ptf import *
from src.h_utils import *
# enable custom summarization
print('libraries loaded')

# Script control parameters
# Set prediction target 'yield, 'product' or both
TARGET = ['yield']
CROPS = ['wheat_winter', 'wheat_spring', 'rice',
         'soy', 'maize']    # set to target crops

PCT_DEBUG = 1      # Pct of dataset used
AREA = 'all'       # ['all', 'france', 'europe', 'china', 'US']
CRUTS_ONLY = True  # ONly use Cruts variables in training set
TRAIN_MIX = True  # Mix simulated and recorded training data

# drop variables with low correlation
CORR_THR = 0.15    # threshold for minimum correlation between feature and target

# Conversion to PCA
CONV_PCA = False             # Use PCA faetures instead of regular
N_PCA = 30                  # Number of PCAs

# Timeseries parameters (total timeseris years <= input_ch + hor_train + hor_test)
RAND_TEST = False    # Train on whole time and use random pixels as testing
PCT_TEST = 0.2      # Pixels used for testing if randomised
INP_LENGTH = 18     # Length of timeseries training period
MAX_PRED_LEN = 4       # Length of prediction period
MAX_ENCODE_LEN = 18     # Length of encoder period

# Model epochs run
EPOCHS = 20
BATCH_S = 64

# Model hyperparameters
HID_DIM = 64
LEARN_RATE = 0.001
LSTM_L = 8
MAX_S = 1
N_HEADS = 4
DROPOUT = 0.1

# Set parameter to use quantile regression (probabilistic) or deterministic loss
PROBABILISTIC = False   # if false, model uses MSELoss as loss_fn
QUANTILES = [0.1, 0.5, 0.9] #[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

# name of model
""" 
tag = 'pft2'
model_name = tag + '_tft_' + ''.join(CROPS) + '_' + \
    ''.join(TARGET) + '_' + AREA + '_corr' + str(CORR_THR)

if CONV_PCA:
    model_name = model_name + '_pca'

 """
model_name = 'test_prediction_1year'

print('parameteres set')

# Data constructor and manipulation classes
data = data_maker(TARGET, CROPS, PCT_DEBUG, AREA, CORR_THR, N_PCA, CRUTS_ONLY)
df_train = data.get_training_data(CONV_PCA, TRAIN_MIX)

print('data loaded')

# Convert from Pandas to PtF ts and PT Dataloaders
training_cutoff = df_train["year"].max() - MAX_PRED_LEN

tsp = timeSeriesPtf(MAX_PRED_LEN, MAX_ENCODE_LEN, training_cutoff)

ts_train, ts_val = tsp.get_training_ts(df_train)

train_load, val_load = tsp.get_train_dataloaders(ts_train, ts_val, BATCH_S)
print('dataloaders set')

# Early stopping condition
early_stopper = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")
model_sum = ModelSummary(max_depth=1)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
#    limit_train_batches=200,  # run validation every x batches
#    limit_val_batches=1000,
    enable_model_summary=True,
    callbacks=[lr_logger, early_stopper, model_sum],
    logger=logger,
    gradient_clip_val=0.1,
)

my_model = TemporalFusionTransformer.from_dataset(
    ts_train,
    learning_rate=LEARN_RATE,
    hidden_size=HID_DIM,
    attention_head_size=N_HEADS,
    dropout=DROPOUT,
    hidden_continuous_size=LSTM_L,
    loss=QuantileLoss(
        QUANTILES
    ),
)

trainer.fit(
    my_model,
    train_dataloaders=train_load,
    val_dataloaders=val_load,
)

# Save the best model according to validation set
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
path = cwd + '/models/' + model_name

model_dict = {'model': best_model, 
                'data_maker': data,
                'tsp': tsp}

pickl_save(model_dict, path)
