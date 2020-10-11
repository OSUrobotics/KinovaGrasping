#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:35:40 2019

@author: orochi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.mlls.variational_elbo import VariationalELBO
import math

class LinearNetwork(nn.Module):

    def __init__(self):
        super(LinearNetwork, self).__init__()
        self.fc1 = nn.Linear(72, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

class LinearNetwork3Layer(nn.Module):
    
    def __init__(self):
        super(LinearNetwork3Layer, self).__init__()
        self.fc1 = nn.Linear(72, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class LinearNetwork4Layer(nn.Module):
    
    def __init__(self):
        super(LinearNetwork4Layer, self).__init__()
        self.fc1 = nn.Linear(72, 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class ReducedLinearNetwork(nn.Module):
    def __init__(self):
        super(ReducedLinearNetwork, self).__init__()
        self.fc1 = nn.Linear(12, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
    
class ReducedLinearNetwork3Layer(nn.Module):
    
    def __init__(self):
        super(ReducedLinearNetwork3Layer, self).__init__()
        self.fc1 = nn.Linear(12, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class ReducedLinearNetwork4Layer(nn.Module):
    
    def __init__(self):
        super(ReducedLinearNetwork4Layer, self).__init__()
        self.fc1 = nn.Linear(12, 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class SmallNetwork(nn.Module):
    def __init__(self):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(15, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x