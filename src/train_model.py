from tkinter import Y
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
from math import sqrt
from src.utils import makeDataset, rmse, get_data
from src.models import kfold_cv, neuralNet, train_epoch, train_k_epochs

import tqdm
import warnings
import torch.optim as optim
import optuna

warnings.filterwarnings("ignore")

# PYTORCH
df_train, df_test = get_data('wheat_winter')

# Set up CUDA
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Cuda Device Available")
    print("Name of the Cuda Device: ", torch.cuda.get_device_name())
    print("GPU Computational Capability: ", torch.cuda.get_device_capability())
else:
    device = 'cpu'

# Hyperparameters
model = neuralNet(df_train.shape[1] - 1).to(device)
target = 'yield'
batch_size = 1024
EPOCHS = 5
learning_rate = 0.001
K = 5

criterion = nn.MSELoss()

optm = Adam(model.parameters(), lr=learning_rate)

dataset_train = makeDataset(df_train, 'yield')
dataset_test = makeDataset(df_test, 'yield')


dl_train = DataLoader(dataset_train, batch_size)
dl_test = DataLoader(dataset_test, batch_size)

train_epoch(dl_train, model, optm, criterion, device)
train_k_epochs(dl_train, dl_test, model, optm, criterion, epochs = EPOCHS, device = device)

kfold = kfold_cv(dataset_train, model, optm, criterion, EPOCHS, batch_size, K, device)
foldperf = kfold


avg = pd.DataFrame({
    "Epoch:": [],
    "loss": [],
    "train_rmse": [],
    "test_rmse": []
})

x = foldperf['fold{}'.format(1)]['loss'].reset_index()['loss'][0]
y = foldperf['fold{}'.format(1)]['loss'].reset_index()['loss'][1]

x + y

for e in range(0, EPOCHS):
    loss, tar, ter = 0, 0, 0
    for k in range(1, K+1):
        loss += foldperf['fold{}'.format(k)]['loss'].reset_index()['loss'][e]
        tar += foldperf['fold{}'.format(k)]['train_rmse'].reset_index()['train_rmse'][e]
        ter += foldperf['fold{}'.format(k)]['test_rmse'].reset_index()['test_rmse'][e]
    print(loss, tar, ter)
    avg_ep = pd.DataFrame({
        "Epoch:": [e + 1],
        "loss": [loss / K],
        "train_rmse": [tar / K],
        "test_rmse": [ter / K]
    })
    avg = pd.concat([avg, avg_ep])


ml = pd.DataFrame({
    "Epoch:": [],
    "loss": [],
    "train_rmse": [],
    "test_rmse": []
})

foldperf = kfold


import matplotlib.pyplot as plt

plt.plot(kfold['fold1']['Epoch:'], kfold['fold1'].train_rmse)
plt.show()



def objective(trial, inp_features, dataset_train, criterion):

    params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              'n_unit': trial.suggest_int("n_unit", 4, 20)
              }
    
    model = neuralNet(inp_features, params['n_unit'])

    optm = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])
    kfold_performance = kfold_cv(dataset_train, model, optm, criterion)
    return kfold_performance

    
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)

target = 'yield'

# Split for rmse values
x_train = df_train.loc[:, df_train.columns != target]
y_train = df_train[target]
x_test = df_test.loc[:, df_test.columns != target]
y_test = df_test[target]

# Train linear model
linr = LinearRegression()
linr.fit(x_train, y_train)

y_hat_linreg = pd.DataFrame(linr.predict(x_test))

x_nn_train = torch.Tensor(x_train.values).to(device)
x_nn_test = torch.Tensor(x_test.values).to(device)

# Calc RMSE
y_hat_nn_test = pd.DataFrame(model(x_nn_test).cpu().detach())
y_hat_nn_train = pd.DataFrame(model(x_nn_train).cpu().detach())

rmse(y_train, y_hat_nn_train)

rmse(y_test, y_hat_linreg)
rmse(y_test, y_hat_nn_test)