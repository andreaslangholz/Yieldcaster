import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from src.utils import get_data, rmse, baseline_rolling
from src.models import makeDataset, neuralNet, bigNeuralNet, train_k_epochs
import warnings
from torch.optim import Adam
from torch.utils.data import DataLoader
 
warnings.filterwarnings("ignore")

# Get data
df_train, df_test = get_data('wheat_winter', type = 'pix')
# Set up CUDA
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print('Device: ', device)

# Pick a model
model_name = 'small_5'
model = neuralNet(df_train.shape[1] - 1)

# Hyperparameters
target = 'yield'
batch_size = 256
EPOCHS = 5
learning_rate = 0.001
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Make datasets
dataset_train = makeDataset(df_train, 'yield')
dataset_test = makeDataset(df_test, 'yield')

train_load = DataLoader(dataset_train, batch_size=batch_size)
test_load = DataLoader(dataset_test, batch_size=batch_size)

#Train model
model_loss = train_k_epochs(train_load, test_load, model, optimizer, criterion, out=True, epochs=EPOCHS, device=device)

# Save the model!
#TODO: Save the model
model_loss.to_csv("C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\outputs\\hist_errors\\" + model_name + "_loss.csv")

### Compare outputs ###
# Predict with model
x_train = df_train.loc[:, df_test.columns!=target]
x_test = df_test.loc[:, df_test.columns!=target]

y_train = df_train[target]
y_test = df_test[target]

model.eval()
with torch.no_grad():
    y_hat_model_train = model(torch.Tensor(x_train.values)).detach().numpy()
    y_hat_model_test = model(torch.Tensor(x_test.values)).detach().numpy()

df_results_train = df_train[['lon', 'lat', 'year', 'yield']]
df_results_test = df_test[['lon', 'lat', 'year', 'yield']]

df_results_train['model_pred'] = y_hat_model_train
df_results_test['model_pred'] = y_hat_model_test

# Predict linreg
linr = LinearRegression().fit(x_train, y_train)

y_hat_linr_train = linr.predict(x_train)
y_hat_linr_test = linr.predict(x_test)

df_results_train['linr_pred'] = y_hat_linr_train
df_results_test['linr_pred'] = y_hat_linr_test

# Predict rolling avg
df_roll_train = baseline_rolling(df_train, 3)
df_roll_test = baseline_rolling(df_test, 3)

df_roll_train.drop('yield', axis=1, inplace = True)
df_roll_test.drop('yield', axis=1, inplace = True)

# Merge and output
df_results_train = df_results_train.merge(df_roll_train, on=['lon', 'lat', 'year'], how = 'inner')
df_results_test = df_results_test.merge(df_roll_test, on=['lon', 'lat', 'year'], how = 'inner')

df_results_train.to_csv("C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\outputs\\hist_errors\\" + model_name + "_err_train.csv")
df_results_test.to_csv("C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\outputs\\hist_errors\\" + model_name + "err_test.csv")

