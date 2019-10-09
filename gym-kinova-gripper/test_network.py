#!/usr/bin/env python

import torch 
from train import train_network
import csv
import pandas as pd
import numpy as np
import pdb
import NCS_nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("heart.csv")
cols = set(data.columns)
cols.remove("sex")
x = np.array(data[cols])
y = np.array(data["sex"])
# pdb.set_trace()

actor_net = NCS_nn.GraspValid_net(13).to(device)

train_network([x, y], actor_net, 2, 200, 10, model_path="test_model")

correct= 0
test_x = x[np.arange(200, 303)]
test_y = y[np.arange(200, 303)]
for i in range(100):
	outputs = actor_net(test_x[i])
	pdb.set_trace()
	# correct += (predicted == test_y)