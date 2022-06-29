import matplotlib.pyplot as plt
import matplotlib
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
from math import sqrt
from src.utils import makeDataset, rmse, neuralNet

datapath = "C:\\Users\\langh\\OneDrive\\Documents\\Imperial\\Individual_project\\Data\\Processed\\"
df_comb_90 = pd.read_csv(
    datapath + "historical_combined_train90.csv", index_col=0)
df_comb_10 = pd.read_csv(
    datapath + "historical_combined_test10.csv", index_col=0)

sys.path

pd.read_csv("C:\\Users\\langh\\OneDrive\\Documents\\Imperial\\Individual_project\\Yieldcaster\\data\\raw\\Berkeley Earth\\Complete_TMAX_Daily_LatLong1_1980.nc")

# Linear model
x_train = df_comb_90[['lon', 'lat', 'year', 'SPEI', 'extHeat', 'tmp', 'cld', 'dtr',
                      'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']]

y_train = df_comb_90['yield']

# Train linear model
linr = LinearRegression()
linr.fit(x_train, y_train)

x_test = df_comb_10[['lon', 'lat', 'year', 'SPEI', 'extHeat', 'tmp', 'cld', 'dtr',
                     'frs', 'pet', 'pre', 'tmn', 'tmx', 'vap', 'wet']]

y_test = df_comb_10['yield']

y_hat_linreg = pd.DataFrame(linr.predict(x_test))


# Hyperparameters
target = 'yield'
batch_size = round(len(df_comb_90) / 100)  # powers of 2
EPOCHS = 30
learning_rate = 0.001
criterion = nn.MSELoss()
optm = Adam(model.parameters(), lr=learning_rate)



# PYTORCH
# Training function
def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss, output


# Set up CUDA
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Cuda Device Available")
    print("Name of the Cuda Device: ", torch.cuda.get_device_name())
    print("GPU Computational Capability: ", torch.cuda.get_device_capability())
else:
    device = 'cpu'

dataset_train = makeDataset(
    datapath + "historical_combined_train90.csv", 'yield')
dataload_train = DataLoader(dataset=dataset_train, num_workers=8,
                            pin_memory=True, batch_size=batch_size, shuffle=False)

x_nn_train = torch.Tensor(x_train.values).to(device)
x_nn_test = torch.Tensor(x_test.values).to(device)

model_loss = pd.DataFrame({
    "Epoch:": [],
    "Training set RMSE": [],
    "Test set RMSE": []
})

model = NN(df_comb_90.shape[1] - 1).to(device)

for epoch in range(EPOCHS):

    epoch_loss = 0
    for batch in tqdm(dataload_train):
        x_train, y_train = batch['inp'], batch['oup']
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        loss, predictions = train(model, x_train, y_train, optm, criterion)
        epoch_loss += loss

    y_hat_nn = pd.DataFrame(model(x_nn_test).cpu().detach())
    rmse_test = rmse(y_test, y_hat_nn)

    print('Epoch {} of {} Loss : {}'.format((epoch + 1), EPOCHS, epoch_loss))
    print('Epoch {} Test set RMSE : {}'.format((epoch + 1), rmse_test))

    ml = pd.DataFrame({
        "Epoch:": [epoch + 1],
        "Training set RMSE": [sqrt(epoch_loss)],
        "Test set RMSE": [rmse_test]
    })

    model_loss = pd.concat([model_loss, ml])

# Calc RMSE
y_hat_nn_test = pd.DataFrame(model(x_nn_test).cpu().detach())
y_hat_nn_train = pd.DataFrame(model(x_nn_train).cpu().detach())

rmse(y_train, y_hat_nn_train)
rmse(y_test, y_hat_linreg)
rmse(y_test, y_hat_nn)

y_hat_nn_zeros = y_hat_nn.clip(lower=0)
rmse(y_test, y_hat_nn_zeros)