import torch
import torch.nn as nn
from torch import autograd
import pandas as pd
import numpy as np


class LinearRegression(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(in_features=n_features, out_features=1)
        # raise NotImplementedError()

    def forward(self, x):
        return self.linear(x)


a = torch.tensor([1., 1., 2.])
print(a.sum())