from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from h_utils import get_data, rmse, baseline_rolling
from models import makeDataset, neuralNet, twoLayerNN, threeLayerNN, train_k_epochs
import warnings
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# Get data
df_train, df_test = get_data('rice', type='pix')
# Set up CUDA
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print('Device: ', device)

# Pick a model
model_name = 'twolayer500x500'
model = twoLayerNN(df_train.shape[1] - 1, L1 = 500, L2 = 300)

# Hyperparameters
target = 'yield'
batch_size = 256
EPOCHS = 30
learning_rate = 0.001
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scaler = MinMaxScaler()

# Make datasets
dataset_train = makeDataset(df_train, target)
dataset_test = makeDataset(df_test, target)

# Normalise features
scaler.fit(df_train.loc[:, df_test.columns != target])
dataset_train.normalise(scaler)
dataset_test.normalise(scaler)

# Make dataloaders
train_load = DataLoader(dataset_train, batch_size=batch_size)
test_load = DataLoader(dataset_test, batch_size=batch_size)

# Train model
model_loss = train_k_epochs(train_load, test_load, model,
                            optimizer, criterion, out=True, epochs=EPOCHS, device=device)

# Save the model
torch.save(
    model, "C:\\Users\\Andreas Langholz\\Yieldcaster\\models\\" + model_name + ".pt")
model_loss.to_csv(
    "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\outputs\\hist_errors\\" + model_name + "_loss.csv")

### Compare outputs ###
# Predict with model
scaler.fit(df_train.loc[:, df_test.columns != target])
x_train = scaler.transform(df_train.loc[:, df_test.columns != target])
x_test = scaler.transform(df_test.loc[:, df_test.columns != target])

y_train = df_train[target]
y_test = df_test[target]

model.eval()
with torch.no_grad():
    y_hat_model_train = model(torch.Tensor(x_train)).detach().numpy()
    y_hat_model_test = model(torch.Tensor(x_test)).detach().numpy()

df_results_train = df_train[['lon', 'lat', 'year', target]]
df_results_test = df_test[['lon', 'lat', 'year', target]]

df_results_train['model_pred'] = y_hat_model_train
df_results_test['model_pred'] = y_hat_model_test

# Predict linreg
linr = LinearRegression().fit(x_train, y_train)

y_hat_linr_train = linr.predict(x_train)
y_hat_linr_test = linr.predict(x_test)

df_results_train['linr_pred'] = y_hat_linr_train
df_results_test['linr_pred'] = y_hat_linr_test

# Predict rolling avg

df_roll_train = baseline_rolling(df_train, 5)
df_roll_test = baseline_rolling(df_test, 5)

df_roll_train.drop('yield', axis=1, inplace=True)
df_roll_test.drop('yield', axis=1, inplace=True)

# Merge and output
df_results_train = df_results_train.merge(
    df_roll_train, on=['lon', 'lat', 'year'], how='inner')
df_results_test = df_results_test.merge(
    df_roll_test, on=['lon', 'lat', 'year'], how='inner')


df_results_test[['model_pred', target]].describe()
rmse(df_results_test[target], df_results_test['model_pred'])
rmse(df_results_test[target], df_results_test['linr_pred'])
rmse(df_results_test[target], df_results_test['rolling_yield3'])


df_results_train.to_csv(
    "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\outputs\\hist_errors\\" + model_name + "_err_train.csv")
df_results_test.to_csv(
    "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\outputs\\hist_errors\\" + model_name + "err_test.csv")