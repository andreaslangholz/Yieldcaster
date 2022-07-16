from sklearn.metrics import mean_squared_error
from numpy import sqrt
from torch.utils.data import Dataset, DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.nn import functional
from tqdm import tqdm
import torch.nn as nn
import torch
import src.utils as ut
from torch.autograd import Variable
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch
import torch.nn as nn
from src.utils import get_data, rmse, baseline_rolling
from src.models import makeDataset, neuralNet, twoLayerNN, threeLayerNN, train_k_epochs
from torch.optim import Adam

class sequenceData(Dataset):
    def __init__(self, df, target='yield'):
        self.df = df
        try:
            if target == 'yield':
                self.df.drop('production', axis=1, inplace=True)
            else:
                self.df.drop('yield', axis=1, inplace=True)
        except:
            pass
        try:
            self.df.drop('index', axis=1, inplace=True)
            self.df.drop('level_0', axis=1, inplace=True)
        except:
            pass
        self.df = self.df.dropna()
        self.pixels = df.groupby(
            ['lon', 'lat'], as_index=False).size().drop('size', axis=1)
        self.len = self.pixels.shape[0]
        self.df = self.df.sort_values(['year', 'lon', 'lat'])
        self.oup = df[['lon', 'lat', target]]
        self.inp = df.loc[:, df.columns != target]
    def normalise(self, scaler):
        self.scaler = scaler
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        ids = self.pixels.loc[idx]
        inpt = self.inp[self.inp['lon'].isin(
            [ids['lon']]) & self.inp['lat'].isin([ids['lat']])]
        oupt = self.oup[self.oup['lon'].isin(
            [ids['lon']]) & self.oup['lat'].isin([ids['lat']])]
        oupt = oupt.drop(['lon', 'lat'], axis=1)
        inpt = self.scaler.transform(inpt)
        inpt = Variable(torch.Tensor(inpt))
        oupt = Variable(torch.Tensor(oupt.values))
        return {'inp': inpt,
                'oup': oupt
                }


class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM1, self).__init__()
        self.num_layers = 1  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=False)  # lst
        self.fc = nn.Linear(hidden_size, 1)  # fully connected last layer
        self.relu = nn.ReLU()
    def forward(self, x_tens):
        # Set up x in size of (time, N, features)
        # make first inputs of H_0 and C_0
        h_0 = Variable(torch.zeros(self.num_layers,
                                   x_tens.size(1),
                                   self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers,
                                   x_tens.size(1),
                                   self.hidden_size))  # internal state
        # Propagate input through lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x_tens, (h_0, c_0))
        out = self.fc(output)  # Final Output linear
        out = self.relu(out)  # final output Relu
        return out, (hn, cn)

# Training 1 batch

def train_batch_seq(model, batch, optimizer, criterion):
    N = batch['inp'].size(0)
    seq = batch['inp'].size(1)
    feat = batch['inp'].size(2)
    x_batch = batch['inp']
    y_batch = batch['oup']
    x_tens = torch.reshape(x_batch, (seq, N, feat))
    y_tens = torch.reshape(y_batch, (seq, N))
    model.zero_grad()
    output = model(x_tens)
    y_out = output.view(-1, N)
    loss = criterion(y_out, y_tens)
    loss.backward()
    optimizer.step()
    return loss, output

def train_epoch_seq(dataload_train, model, optimizer, criterion):
    epoch_loss = 0
    nb = 0
    for batch in dataload_train:
        loss, _ = train_batch_seq(
            model, batch, optimizer, criterion)
        epoch_loss += loss
        nb += 1
    loss = epoch_loss / nb
    return loss


def calc_loss_rmse_seq(model, dataloader, device="cpu"):
    se = 0
    Ne = 0
    for batch in dataloader:
        N = batch['inp'].size(0)
        seq = batch['inp'].size(1)
        feat = batch['inp'].size(2)
        x_batch = batch['inp']
        y_tens = batch['oup']
        x_tens = torch.reshape(x_batch, (seq, N, feat))
#        y_tens = torch.reshape(y_batch, (seq, N))
        b = N * seq
        Ne += b
        predictions = model(x_tens)
        y = y_tens.view(-1, N)
        y = y.detach().numpy()
        y_hat = predictions.view(-1, N)
        y_hat = y_hat.detach().numpy()
        se += b * mean_squared_error(y, y_hat)
    rmse = sqrt(se / Ne)
    return rmse, se


def train_k_epochs_seq(train_load, test_load, model, optimizer, criterion, out=False, epochs=50, device='cpu', delta_break=10e-5):
    if out:
        model_loss = pd.DataFrame({
            "Epoch:": [],
            "loss": [],
            "train_rmse": [],
            "test_rmse": []
        })
    else:
        model_loss = pd.DataFrame({
            "Epoch:": [],
            "loss": []
            })
    last_loss = 1
    for epoch in tqdm(range(epochs)):
        epoch_loss = train_epoch_seq(
            train_load, model, optimizer, criterion)
        if out:
            model.eval()
            with torch.no_grad():
                train_rmse, _ = calc_loss_rmse_seq(model, train_load)
                test_rmse, _ = calc_loss_rmse_seq(model, test_load)
                if out:
                    print('\nEpoch {} of {} MSE : {}'.format(
                        (epoch + 1), epochs, round(epoch_loss.item(), 4)))
                    print('Epoch {} Train set RMSE : {}'.format(
                        (epoch + 1), round(train_rmse, 4)))
                    print('Epoch {} Test set RMSE : {}'.format(
                        (epoch + 1), round(test_rmse, 4)))
                ml = pd.DataFrame({
                    "Epoch:": [epoch + 1],
                    "loss": [epoch_loss.detach().numpy()],
                    "train_rmse": [train_rmse],
                    "test_rmse": [test_rmse]
                })
                model_loss = pd.concat([model_loss, ml])
            model.train()
        else:
                ml = pd.DataFrame({
                    "Epoch:": [epoch + 1],
                    "loss": [epoch_loss.detach().numpy()],
                    "train_rmse": [train_rmse],
                    "test_rmse": [test_rmse]
                })
            
            
        model_loss = pd.DataFrame({
            "Epoch:": [],
            "loss": []
            })

        if abs(epoch_loss - last_loss) < delta_break:
            print('Model converged to less than {} diff. loss'.format(delta_break))
            print('This epoch: {}, last epoch: {}'.format(epoch_loss, last_loss))
            break
        last_loss = epoch_loss
    return model_loss


# Get data
df_train, df_test = ut.get_data('rice', type='pix', all_years=True)
# Set up CUDA
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print('Device: ', device)

# Pick a model
model_name = 'lstm'
model = LSTM1(df_train.shape[1] - 1, 64)

# Hyperparameters
target = 'yield'
batch_size = 256
EPOCHS = 30
learning_rate = 0.001
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scaler = MinMaxScaler()

# Make datasets
dataset_train = sequenceData(df_train, target)
dataset_test = sequenceData(df_test, target)

# Normalise features
scaler.fit(df_train.loc[:, df_train.columns != target])
dataset_train.normalise(scaler)
dataset_test.normalise(scaler)

# Make dataloaders
train_load = DataLoader(dataset_train, batch_size=batch_size)
test_load = DataLoader(dataset_test, batch_size=batch_size)

dataset_test.__getitem__(0)

batch = next(iter(train_load))

N = batch['inp'].size(0)
seq = batch['inp'].size(1)
feat = batch['inp'].size(2)

train_batch_seq(model, batch, optimizer, criterion)

train_epoch_seq(train_load, model, optimizer, criterion)

train_out_lstm = train_k_epochs_seq(train_load, test_load, model, optimizer,criterion, out = True, epochs = 30)

df = df_train
pixels = df.groupby(
    ['lon', 'lat'], as_index=False).size().drop('size', axis=1)
len = pixels.shape[0]
df = df.sort_values(['year', 'lon', 'lat'])
oup = df[['lon', 'lat', target]]
inp = df.loc[:, df.columns != target]

ids = pixels.loc[0]
print(idx)
print(ids)
ids['lon']

inpt = inp[inp['lon'].isin([ids['lon']]) & inp['lat'].isin([ids['lat']])]

oupt = oup[oup['lon'].isin([ids['lon']]) & oup['lat'].isin([ids['lat']])]
oupt = oupt.drop(['lon', 'lat'], axis=1)

{'inp': inpt,
 'oup': oupt
 }


# Train model
model_loss = train_k_epochs_seq(train_load, test_load, model,
                                optimizer, criterion, out=True, epochs=EPOCHS, device=device)


dataset_train = sequenceData(df_train, target)

idx = [1, 2, 3, 11, 1234, 34, 643]
ds = dataset_train.__getitem__(idx)
x_train = ds['inp']
y_train = ds['oup']
years = x_train['year'].nunique()

train_batch_seq(lstm, x_train, y_train, optimizer, criterion)

train_load = DataLoader(dataset_train, batch_size=128)


train_epoch_seq(dataload_train, model, optimizer, criterion, device)


N = x_train.groupby(['lon', 'lat']).count().shape[0]

lstm = LSTM1(num_features, 64)

