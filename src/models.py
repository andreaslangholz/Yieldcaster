from torch.utils.data import Dataset, DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.nn import functional
from tqdm import tqdm
import torch.nn as nn
import warnings
import src.utils as ut
warnings.filterwarnings("ignore")

# Model class


class neuralNet(nn.Module):
    def __init__(self, num_feat_inp, L1=10):
        super().__init__()
        self.hidden_layer = nn.Linear(num_feat_inp, L1)
        self.output_layer = nn.Linear(L1, 1)

    def forward(self, x):
        x = functional.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

## Trainer functions ##

# Training 1 batch


def train_batch(model, x_batch, y_batch, optimizer, criterion):
    model.zero_grad()
    output = model(x_batch)
    loss = criterion(output, y_batch)
    loss.backward()
    optimizer.step()
    return loss, output

# Training 1 epoch


def train_epoch(dataload_train, model, optimizer, criterion, device):
    epoch_loss = 0
    for batch in dataload_train:
        x_train, y_train = batch['inp'], batch['oup']
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        loss, predictions = train_batch(
            model, x_train, y_train, optimizer, criterion)
        epoch_loss += loss
    return epoch_loss


def calc_loss(model, dataloader, loss_fn, device="cpu"):
    loss = 0
    model.eval()
    for batch in dataloader:
        x_train, y_train = batch['inp'], batch['oup']
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        predictions = model(x_train)
        loss += loss_fn(y_train.detach().numpy(), predictions.detach().numpy())
    return loss


def train_k_epochs(train_load, test_load, model, optimizer, criterion, out = False, epochs=50, device='cpu', delta_break=10e-4):
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

        train_rmse = calc_loss(model, train_load, ut.rmse)
        test_rmse = calc_loss(model, test_load, ut.rmse)
        if out: 
            print('\n Epoch {} of {} Loss : {}'.format(
                (epoch + 1), epochs, epoch_loss))
            print('Epoch {} Train set RMSE : {}'.format((epoch + 1), train_rmse))
            print('Epoch {} Test set RMSE : {}'.format((epoch + 1), test_rmse))

        ml = pd.DataFrame({
            "Epoch:": [epoch + 1],
            "loss": [epoch_loss.detach().numpy()],
            "train_rmse": [train_rmse],
            "test_rmse": [test_rmse]
        })

        model_loss = pd.concat([model_loss, ml])

        if abs(epoch_loss - last_loss) < delta_break:
            break

        last_loss = epoch_loss
    return model_loss

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def kfold_cv(dataset, model, optimizer, criterion, epochs=30, batch_size=256, K=5, device="cpu"):
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
            train_load, val_load, mf, optimizer, criterion, epochs, device)

        foldperf['fold{}'.format(fold + 1)] = model_loss
        print('\n')

    avg = pd.DataFrame({"Epoch:": [], "loss": [], "train_rmse": [], "test_rmse": []})

    for e in range(0, epochs):
        loss, tar, ter = 0, 0, 0
        for k in range(1, K+1):
            loss += foldperf['fold{}'.format(k)]['loss'].reset_index()['loss'][e]
            tar += foldperf['fold{}'.format(k)]['train_rmse'].reset_index()['train_rmse'][e]
            ter += foldperf['fold{}'.format(k)]['test_rmse'].reset_index()['test_rmse'][e]
        
        avg_ep = pd.DataFrame({"Epoch:": [e + 1], "loss": [loss / K], 
            "train_rmse": [tar / K], "test_rmse": [ter / K]})

        avg = pd.concat([avg, avg_ep])

    foldperf['avg'] = avg

    return foldperf
