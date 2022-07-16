from src.lstm import sequenceData
from re import X
from tkinter import Variable
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from src.utils import get_data, rmse, baseline_rolling
from src.models import makeDataset, neuralNet, twoLayerNN, threeLayerNN, train_k_epochs
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
model_name = 'twolayer'
model = threeLayerNN(df_train.shape[1] - 1)

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


train_load = DataLoader(dataset_train, batch_size=batch_size)

df_train, df_test = get_data('rice', type='pix', all_years=True)

dataset_train = sequenceData(df_train, target)

idx = [1, 2, 3]
ds = dataset_train.__getitem__(idx)
dsi = ds['inp']
dsi.columns

dsi.drop(['index', 'level_0'], axis=1, inplace=True)
years = dsi['year'].nunique()
len(idx)

df_train.groupby(['lon', 'lat']).count().shape[0]

.nunique()

(years, len(idx), dsi.shape[1])

dsi.size

dsi.shape[1]

dsixx = dsi.sort_values(['year', 'lon', 'lat'])
dsitens = Variable(torch.Tensor(dsixx.values))
dsix = torch.reshape(dsitens, (years, len(idx), dsi.shape[1]))
dsix[:, 1, 0]


lstm = LSTM1(dsi.shape[1], 10, years)
out, hn, cn = lstm(dsix)
out.size()
hn.size()

hn = hn.view(-1, 10)
hn.size()
fc = nn.Linear(10, 1)
relu = nn.ReLU()
out = fc(out)  # Final Output
out = relu(out)  # relu
out.size()

class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM1, self).__init__()
        self.num_layers = 1  # number of layers
        self.input_size = input_size  # input size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=False)  # lst
        self.fc = nn.Linear(hidden_size, 1)  # fully connected last layer
        self.relu = nn.ReLU()
    def forward(self, x_train):
        # Set up x in size of (time, N, features)
        N = x_train.groupby(['lon', 'lat']).count().shape[0]
        x_train = x_train.sort_values(['year', 'lon', 'lat'])
        x_tens = Variable(torch.Tensor(x.values))
        x_tens = torch.reshape(x_tens, (years, N, dsi.shape[1]))        
        # make first inputs of H_0 and C_0
        h_0 = Variable(torch.zeros(self.num_layers,
                                   x.size(1),
                                   self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers,
                                   x.size(1),
                                   self.hidden_size))  # internal state
        
        # Propagate input through lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        
        out = self.fc(output)  # Final Output linear
        out = self.relu(out)  # final output Relu
        return output


class LSTM(nn.Module):
    def __init__(self, num_features, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        self.lstm1 = nn.LSTM(num_features, self.hidden_layers)
        self.linear = nn.LeakyReLU(self.hidden_layers, 1)

    def forward(self, x_train):
        outputs, n_samples = x_train.shape[0]

        h_0 = torch.zeros(self.hidden_layers, n_samples, dtype=torch.float32)
        c_0 = torch.zeros(self.hidden_layers, n_samples, dtype=torch.float32)

        time = x_train['year'].unique()
        for year in time:
            x_train_t = x_train[x_train['year'] == year]
            h_t, c_t = self.lstm1(x_train_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)

        # transform list to tensor
        outputs = torch.cat(outputs, dim=1)
        return outputs


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
df_roll_train = baseline_rolling(df_train, 3)
df_roll_test = baseline_rolling(df_test, 3)

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
