import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from src.utils import makeDataset, rmse, get_data
from src.models import kfold_cv, neuralNet, train_epoch, train_k_epochs, objective
import warnings
import optuna
from torch.optim import Adam

warnings.filterwarnings("ignore")

# Get data
df_train, _ = get_data('wheat_winter')

# Set up CUDA
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print('Device: ', device)

# Pick a model
model = neuralNet(df_train.shape[1] - 1)

# Hyperparameters
target = 'yield'
batch_size = 1024
EPOCHS = 5
learning_rate = 0.001
K = 5
optimiser = Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Make datasets
dataset_train = makeDataset(df_train, 'yield')

kfold = kfold_cv(dataset_train, model, optimiser, criterion,
                 EPOCHS, batch_size, K, device)

kfold['avg'].to_csv("C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\ \
    outputs\\model_performance\\model_perf1.csv")
