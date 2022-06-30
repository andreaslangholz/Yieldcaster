import torch
import pandas as pd
import torch.nn as nn
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
import torch.nn.functional as functional
import re


class Boston_Dataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        self.data = torch.from_numpy(self.df.drop(['target'], axis=1).values)
        self.targets = torch.from_numpy(self.df['target'].values)
        return self.data[idx], self.targets[idx].item()

    def __len__(self):
        return len(self.targets)

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

def rmse(actual, pred):
    return sqrt(mean_squared_error(actual, pred))

def maximumZero(x):
    if x > 0:
        return x
    else:
        return 0

def timeToDays(timestr):
    if type(timestr) == str:
        days_span = re.search("days", timestr)
        days_str = timestr[:days_span.span()[0] - 1]
        hour_str = timestr[days_span.span()[1] + 1:days_span.span()[1] + 3]
        return int(days_str) if hour_str == '' else int(days_str) + round(int(hour_str) / 24)
    else:
        return int(timestr)
