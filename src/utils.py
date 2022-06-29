import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as functional
import re

class makeDataset(Dataset):
    def __init__(self, csvpath, target, mode='train'):
        self.mode = mode
        df = pd.read_csv(csvpath, index_col=0)

        if self.mode == 'train':
            df = df.dropna()
            self.oup = df.pop(target).values.reshape(len(df), 1)
            self.inp = df.values
        else:
            self.inp = df.values

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        if self.mode == 'test':
            inpt = torch.Tensor(self.inp[idx])
            oupt = torch.Tensor(self.oup[idx])
            return {'inp': inpt,
                    'oup': oupt
                    }
        else:
            inpt = torch.Tensor(self.inp[idx])
            return {'inp': inpt
                    }

# Model class
class neuralNet(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.hidden_layer = nn.Linear(num_feat, 10)
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        x = functional.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

def rmse(actual, pred):
    return sqrt(mean_squared_error(actual, pred))

def maximumZero(x):
    if x > 0:
        return x
    else:
        return 0

def timeToDays(timestr):
    days_span = re.search("days", timestr)
    days_str = timestr[:days_span.span()[0] - 1]
    hour_str = timestr[days_span.span()[1] + 1:days_span.span()[1] + 3]
    return int(days_str) if hour_str == '' else int(days_str) + round(int(hour_str) / 24)
