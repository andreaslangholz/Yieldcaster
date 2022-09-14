import os
import sys
import pytorch_forecasting.models.temporal_fusion_transformer.tuning as pft_tune
import warnings
warnings.filterwarnings('ignore')
cwd = os.getcwd()
sys.path.insert(0, cwd)
from src.h_utils import *
from src.OLD.h_ptf import *

# Script control parameters
# Set prediction target 'yield, 'product' or both
TARGET = ['yield']
CROPS = ['wheat_winter', 'wheat_spring', 'rice', 'soy', 'maize']    # set to target crops
PCT_DEBUG = 1     # Pct of dataset used
AREA = 'all'       # ['all', 'france', 'europe', 'china', 'US']
CRUTS_ONLY = True  # ONly use Cruts variables in training set
TRAIN_MIX = True  # Mix simulated and recorded training data

# drop variables with low correlation
D_CORR = True
CORR_THR = 0.1    # threshold for minimum correlation between feature and target

# Conversion to PCA
CONV_PCA = False             # Use PCA faetures instead of regular
N_PCA = 30                  # Number of PCAs

# Load model or run new
LOAD = False

# Timeseries parameters (total timeseris years <= input_ch + hor_train + hor_test)
RAND_TEST = False    # Train on whole time and use random pixels as testing
PCT_TEST = 0.2      # Pixels used for testing if randomised
MAX_PRED_LEN = 2
MAX_ENCODE_LEN = 18

# Model epochs run
EPOCHS = 30
BATCH_S = 32

# name of model
tag = 'all'
model_name = 'optuna_study_new'
model_path = cwd + "/models/" + model_name

if CONV_PCA:
    model_name = model_name + '_pca'

# Data constructor and manipulation classes
data = data_maker(TARGET, CROPS, PCT_DEBUG, AREA, CORR_THR, N_PCA, CRUTS_ONLY)
df_train = data.get_training_data(CONV_PCA, TRAIN_MIX)

# Convert from Pandas to PtF ts and PT Dataloaders
training_cutoff = df_train["year"].max() - MAX_PRED_LEN

tsp = timeSeriesPtf(MAX_PRED_LEN, MAX_ENCODE_LEN, training_cutoff)
ts_train, ts_val = tsp.get_training_ts(df_train)
train_load, val_load = tsp.get_train_dataloaders(ts_train, ts_val, BATCH_S)

# create study
param_study = pft_tune.optimize_hyperparameters(
    train_load,
    val_load,
    model_path=cwd + "/models/ptf_hyperparam/",
    n_trials=30,
    max_epochs=30,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=200, limit_val_batches=50),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,
)

model_dict = {'model': param_study, 
                'data_maker': data,
                'tsp': tsp}

# save study results - also we can resume tuning at a later point in time
path = cwd + "/models/ptf_hyperparam/params_" + model_name
pickl_save(model_dict, path)

# show best hyperparameters
print(param_study.best_trial.params)


