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
import src.h_utils as ut
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch
import torch.nn as nn
from src.h_utils import get_data, rmse, baseline_rolling
from src.h_darts import makeDataset, neuralNet, twoLayerNN, threeLayerNN, train_k_epochs
from torch.optim import Adam

### PYTORCH PIPELINE ##
# Helper functions to run on the Pytorch based pipeline
# OBS: Main model for prediction is using the alternative 
# DARTS API implementation (see below)


# Model classes
class neuralNet(nn.Module):
    def __init__(self, num_feat_inp, L1=10):
        super().__init__()
        self.hidden_layer = nn.Linear(num_feat_inp, L1)
        self.output_layer = nn.Linear(L1, 1)

    def forward(self, x):
        x = functional.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

class twoLayerNN(nn.Module):
    def __init__(self, num_feat_inp, L1=50, L2=10):
        super().__init__()
        self.hidden_l1 = nn.Linear(num_feat_inp, L1)
        self.hidden_l2 = nn.Linear(L1, L2)
        self.output_layer = nn.Linear(L2, 1)

    def forward(self, x):
        x = functional.relu(self.hidden_l1(x))
        x = functional.leaky_relu(self.hidden_l2(x))
        x = functional.leaky_relu(self.output_layer(x))
        return x

class threeLayerNN(nn.Module):
    def __init__(self, num_feat_inp, L1=50, L2=50, L3=50):
        super().__init__()
        self.hidden_l1 = nn.Linear(num_feat_inp, L1)
        self.hidden_l2 = nn.Linear(L1, L2)
        self.hidden_l3 = nn.Linear(L2, L3)
        self.output_layer = nn.Linear(L3, 1)

    def forward(self, x):
        x = functional.leaky_relu(self.hidden_l1(x))
        x = functional.tanh(self.hidden_l2(x))
        x = functional.tanh(self.hidden_l3(x))
        x = functional.leaky_relu(self.output_layer(x))
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.num_layers = 1  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)  # lst
        self.fc = nn.Linear(hidden_size, 1)  # fully connected last layer
        self.relu = nn.ReLU()
    def forward(self, x_tens):
        # Set up x in size of (time, N, features)
        # make first inputs of H_0 and C_0
        h_0 = Variable(torch.zeros(self.num_layers,
                                   x_tens.size(0),
                                   self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers,
                                   x_tens.size(0),
                                   self.hidden_size))  # internal state
        # Propagate input through lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x_tens, (h_0, c_0))
        out = self.fc(output)  # Final Output linear
        out = self.relu(out)  # final output Relu
        return out, (hn, cn)


## Trainer functions ##

# Training 1 batch
def train_batch(model, x_batch, y_batch, optimizer, criterion):
    optimizer.zero_grad()
    output = model(x_batch)
    loss = criterion(output, y_batch)
    loss.backward()
    optimizer.step()
    return loss, output

# Training 1 epoch

def train_epoch(dataload_train, model, optimizer, criterion, device):
    epoch_loss = 0
    nb = 0
    for batch in dataload_train:
        x_train, y_train = batch['inp'], batch['oup']
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        loss, _ = train_batch(
            model, x_train, y_train, optimizer, criterion)
        epoch_loss += loss
        nb += 1
    loss = epoch_loss / nb
    return loss


def calc_loss_rmse(model, dataloader, device="cpu"):
    se = 0
    N = 0
    for batch in dataloader:
        x_train, y_train = batch['inp'], batch['oup']
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        b = len(x_train)
        N += b
        predictions = model(x_train)
        se += b * \
            mean_squared_error(y_train.detach().numpy(),
                               predictions.detach().numpy())

    rmse = sqrt(se / N)
    return rmse, se


def criterion_rmse(input, target):
    y_hat = input.detach().numpy()
    y = target.detach().numpy()
    rmse = np.array(sqrt(mean_squared_error(y_hat, y)))
    return torch.Tensor(rmse)


def train_k_epochs(train_load, test_load, model, optimizer, criterion, out=False, epochs=50, device='cpu', delta_break=10e-5):
    model_loss = pd.DataFrame({
        "Epoch:": [],
        "loss": [],
        "train_rmse": [],
        "test_rmse": []
    })

    last_loss = 1

    for epoch in tqdm(range(epochs)):

        epoch_loss = train_epoch(
            train_load, model, optimizer, criterion, device)

        model.eval()

        with torch.no_grad():
            train_rmse, _ = calc_loss_rmse(model, train_load)
            test_rmse, _ = calc_loss_rmse(model, test_load)
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

        if abs(epoch_loss - last_loss) < delta_break:
            print('Model converged to less than {} diff. loss'.format(delta_break))
            print('This epoch: {}, last epoch: {}'.format(epoch_loss, last_loss))
            break
        last_loss = epoch_loss

        model.train()

    return model_loss


def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def kfold_cv(dataset, model, optimizer, criterion, out=False, epochs=30, batch_size=256, K=5, device="cpu"):
    splits = KFold(n_splits=K, shuffle=True, random_state=42)
    foldperf = {}

    for fold, (train_idx, val_idx) in tqdm(enumerate(splits.split(np.arange(len(dataset))))):
        mf = model.copy()

        print('\nFold {}\n'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_load = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler)
        val_load = DataLoader(
            dataset, batch_size=batch_size, sampler=val_sampler)

        mf.to(device)

        model_loss = train_k_epochs(
            train_load, val_load, mf, optimizer, criterion, out, epochs, device)

        foldperf['fold{}'.format(fold + 1)] = model_loss
        print('\n')

    avg = pd.DataFrame(
        {"Epoch:": [], "loss": [], "train_rmse": [], "test_rmse": []})

    # Calculate the averages of the folds
    for e in range(0, epochs):
        loss, tar, ter = 0, 0, 0
        for k in range(1, K+1):
            loss += foldperf['fold{}'.format(k)
                             ]['loss'].reset_index()['loss'][e]
            tar += foldperf['fold{}'.format(k)
                            ]['train_rmse'].reset_index()['train_rmse'][e]
            ter += foldperf['fold{}'.format(k)
                            ]['test_rmse'].reset_index()['test_rmse'][e]

        avg_ep = pd.DataFrame({"Epoch:": [e + 1], "loss": [loss / K],
                               "train_rmse": [tar / K], "test_rmse": [ter / K]})

        avg = pd.concat([avg, avg_ep])

    foldperf['avg'] = avg

    return foldperf

class makeDataset(Dataset):
    def __init__(self, df, target, mode='train'):
        self.mode = mode
        self.df = df
        if self.mode == 'train':
            self.df = self.df.dropna()
            self.oup = self.df.pop(target).values.reshape(len(df), 1)
            self.inp = self.df.values
        else:
            self.inp = self.df.values

    def normalise(self, scaler):
        self.inp = scaler.transform(self.inp)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt = torch.Tensor(self.inp[idx])
            oupt = torch.Tensor(self.oup[idx])
            return {'inp': inpt,
                    'oup': oupt
                    }
        else:
            inpt = torch.Tensor(self.inp[idx])
            return {'inp': inpt
                    }

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

# Training 1 batch

def train_batch_seq(model, batch, optimizer, criterion):
    N = batch['inp'].size(0)
    seq = batch['inp'].size(1)
    x_batch = batch['inp']
    y_batch = batch['oup']
    y_batch = y_batch.view(-1, seq)
    optimizer.zero_grad()
    output, (hn, cn) = model(x_batch)
    y_out = output.view(-1, seq)
    loss = criterion(y_out, y_batch)
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
        b = N * seq
        Ne += b
        x_batch = batch['inp']
        y_batch = batch['oup']
        y_batch = y_batch.view(-1, seq)
        y = y_batch.detach().numpy()
#        y_tens = torch.reshape(y_batch, (seq, N))
        output, _ = model(x_batch)
        y_hat = output.view(-1, seq)
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

model = LSTM1(df_train.shape[1] - 1, 64)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

batch_iter = iter(train_load)
batch = next(batch_iter)

train_batch_seq(model, batch, optimizer, criterion)

train_epoch_seq(train_load, model, optimizer, criterion)

train_out_lstm = train_k_epochs_seq(train_load, test_load, model, 
    optimizer, criterion, out = True, epochs = 10)


# Train model
model_loss = train_k_epochs_seq(train_load, test_load, model,
                                optimizer, criterion, out=True, epochs=EPOCHS, device=device)

