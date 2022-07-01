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

def df_mth_to_year(df):
    df_var_horizontal = df.sort_values(by=['lon', 'lat', 'year', 'month'])
    df_var_horizontal.drop('month', axis=1, inplace=True)
    df_var_horizontal.drop('time', axis=1, inplace=True)
    df_var_horizontal = (df_var_horizontal.set_index(['lon', 'lat', 'year',
                                                      df_var_horizontal.groupby(['lon', 'lat', 'year']).cumcount().add(
                                                          1)])
                         .unstack()
                         .sort_index(axis=1, level=1))
    df_var_horizontal.columns = [f'{a}{b}' for a, b in df_var_horizontal.columns]
    df_var_horizontal = df_var_horizontal.reset_index()
    return df_var_horizontal
