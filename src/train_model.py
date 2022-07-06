import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
from math import sqrt
from src.utils import makeDataset, rmse, get_data
from src.models import neuralNet, train_epoch
import warnings
warnings.filterwarnings("ignore")

df_train, df_test = get_data('wheat')
target = 'yield'

#TODO: (1) FIX MISSING FRS NANS! (2) fix autoload of index
try:
    for i in range(1,13):
        df_train = df_train.loc[:, df_train.columns != ('frs' + str(i))]
        df_test  = df_test.loc[:, df_test.columns != ('frs' + str(i))]
except:
    print('no frs')

# Split for rmse values
x_train = df_train.loc[:, df_train.columns != target]
y_train = df_train[target]
x_test = df_test.loc[:, df_test.columns != target]
y_test = df_test[target]

# Train linear model
linr = LinearRegression()
linr.fit(x_train, y_train)

y_hat_linreg = pd.DataFrame(linr.predict(x_test))

# PYTORCH

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
EPOCHS = 30
learning_rate = 0.001
criterion = nn.MSELoss()
optm = Adam(model.parameters(), lr=learning_rate)
K = 10
delta_loss_break = 1.e-3

dataset_train = makeDataset(df_train, 'yield')
dataload_train = DataLoader(dataset=dataset_train, num_workers=8,
                            pin_memory=True, batch_size=batch_size, shuffle=False)

x_nn_train = torch.Tensor(x_train.values).to(device)
x_nn_test = torch.Tensor(x_test.values).to(device)

model_loss = pd.DataFrame({
    "Epoch:": [],
    "Training set RMSE": [],
    "Test set RMSE": []
})

last_loss = 1
for epoch in range(EPOCHS):
    epoch_loss = train_epoch(dataload_train, model, optm, criterion, device)

    y_hat_train_nn = pd.DataFrame(model(x_nn_train).cpu().detach())
    y_hat_test_nn = pd.DataFrame(model(x_nn_test).cpu().detach())

    rmse_train = rmse(y_train, y_hat_train_nn)
    rmse_test = rmse(y_test, y_hat_test_nn)

    print('Epoch {} of {} Loss : {}'.format((epoch + 1), EPOCHS, epoch_loss))
    print('Epoch {} Train set RMSE : {}'.format((epoch + 1), rmse_train))
    print('Epoch {} Test set RMSE : {}'.format((epoch + 1), rmse_test))

    ml = pd.DataFrame({
        "Epoch:": [epoch + 1],
        "Training set Loss - Pytorch": [sqrt(epoch_loss)],
        "Training set RMSE - Own": [rmse_train],
        "Test set RMSE": [rmse_test]
    })

    model_loss = pd.concat([model_loss, ml])
    if abs(epoch_loss - last_loss) < delta_loss_break:
        break

    last_loss = epoch_loss

# Calc RMSE
y_hat_nn_test = pd.DataFrame(model(x_nn_test).cpu().detach())
y_hat_nn_train = pd.DataFrame(model(x_nn_train).cpu().detach())

rmse(y_train, y_hat_nn_train)

rmse(y_test, y_hat_linreg)
rmse(y_test, y_hat_nn_test)

rmse(y_test, y_hat_nn_zeros)
x_nn_test.col
df_test

model_errors = pd.DataFrame({
    'lon': df_test.lon,
    'lat': df_test.lat,
    "Test values:": df_test['yield'],
    "pred_nn": y_hat_nn_test,
    "pred_linreg": y_hat_linreg,
})