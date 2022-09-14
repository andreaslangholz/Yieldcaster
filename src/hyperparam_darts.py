""" 
MSc. Project 2022 - Andreas Langholz - Hyperparameter tuning script
""" 

import sys
import os

cwd = os.getcwd()
sys.path.insert(0, cwd)
from src.h_utils import *
from src.h_darts import *
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

#### MODEL CONTROL PARAMETERS
# Set prediction target 'yield, 'product' or both
TARGET = ['yield']
CROPS = ['wheat_winter', 'wheat_spring', 'rice', 'maize', 'soy']    # set to target crops
PCT_DEBUG = 1     # Pct of dataset used
AREA = 'all'      # ['all', 'france', 'europe', 'china', 'US']

# drop variables with low correlation
CORR_THR = 0    # threshold for minimum correlation between feature and target

# Conversion to PCA
CONV_PCA = False             # Use PCA faetures instead of regular
N_PCA = 30                  # Number of PCAs

# Timeseries parameters (total timeseris years <= input_ch + hor_train + hor_test)
HOR_TRAIN = 10       # Length of forecasting period for training
HOR_TEST = 5        # Length of forecasting period for testing


#### HYPERPARAM SEARCH SETTINGS
# Number of combinations to try
num_samples = 100

# Hyperparameter space to search over
config = {
    "batch_size": tune.choice([16, 32, 64, 128]), # First run setting batchsize at 16
    "hidden_size": tune.choice([64, 128, 256, 512]),
    "lstm_layers": tune.choice([1,2,3]),
    "num_attention_heads": tune.choice([1,2,4]), 
    "dropout": tune.uniform(0, 0.3),
    "optimizer_kwargs":{'lr': tune.loguniform(1e-4, 1e-2)}
}

# Early stop callback
my_stopper = EarlyStopping(
    monitor="val_MeanSquaredError",
    patience=3,
    min_delta=0.05,
    mode='min',
)

#### Run the hyper parameter search

# Data constructor and timeseries
data = data_maker(TARGET, CROPS, PCT_DEBUG, AREA, CORR_THR, N_PCA, type='etccdi', all_years=True, encode=True)
df_train = data.get_training_data(CONV_PCA)
tsd = timeseries_darts(HOR_TRAIN, CROPS)
ts_target_train, ts_target_val, ts_cov_train, ts_cov_val = tsd.get_training_ts(df_train)

# Model training function for passing to the optimiser, creates the model using model_args from Ray Tune
def train_model(model_args, callbacks, train, val, cov_train, cov_val):
    torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError(), MeanSquaredError()])
    model = TFTModel(
        input_chunk_length=HOR_TRAIN,
        output_chunk_length=HOR_TEST,
        n_epochs=30,
        torch_metrics=torch_metrics,
        pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": False},
        **model_args
        )

    model.fit(train,
        future_covariates=cov_train, 
        val_series=val,
        val_future_covariates=cov_val, 
        verbose=False, 
        max_samples_per_ts=1, 
        num_loader_workers=4,
    )

# set up ray tune callback and reports
tune_callback = TuneReportCallback(
    {
        "MSE": "val_MeanSquaredError",
        "MAE": "val_MeanAbsoluteError",
        "MAPE": "val_MeanAbsolutePercentageError",
    },
    on="validation_end",
)

reporter = CLIReporter(
    parameter_columns=list(config.keys()),
    metric_columns=['MSE', 'MAE', "MAPE", "training_iteration"],
)

resources_per_trial = {"cpu": 4}

scheduler = ASHAScheduler(max_t=1000, grace_period=3, reduction_factor=2)

train_fn_with_parameters = tune.with_parameters(
    train_model, callbacks=[my_stopper, tune_callback], train = ts_target_train, val = ts_target_val, cov_train = ts_cov_train,cov_val = ts_cov_val
)

# Run analysis 
analysis = tune.run(
    train_fn_with_parameters,
    resources_per_trial=resources_per_trial,
    metric="MSE", 
    mode="min",
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter,
    name="tune_darts",
)

print("Best hyperparameters found were: ", analysis.best_config)

path = cwd + '/models/dartshyperparam'

pickl_save(analysis, path)