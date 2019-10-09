#!/usr/bin/env python

import torch 
import numpy as np
import pickle
import os, sys
import NCS_nn
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file = open("data_cube_5_10_07_19_1612_grasp_10_08_19_0251.pkl", "rb")
data = pickle.load(file)

states = torch.FloatTensor(np.array(data["states"][150000:200000]))
labels = torch.FloatTensor(np.array(data["grasp_sucess"][150000:200000]))

# pdb.set_trace()
model_file = "data_cube_5_grasp_classfier_10_09_19_0309.pt"
model = torch.load(model_file)
grasp_net = NCS_nn.GraspValid_net(47)
grasp_net.load_state_dict(model)
grasp_net.eval()
count = 0
for i in range(50000):

	output = grasp_net(states[i])
	if output >= 0.5 and labels[i] == 1:
		count += 1
	if output < 0.5 and labels[i] == 0:
		count += 1

print("Accuracy", count / 50000)

