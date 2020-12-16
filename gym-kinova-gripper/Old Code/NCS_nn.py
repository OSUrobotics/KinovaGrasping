#!/usr/bin/env python3

'''
Author : Yi Herng Ong
Purpose : Create neural networks for classifiers and controllers

'''
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import TD3
import gym
import utils
import argparse
import torch.optim as optim
import pdb
import pickle
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Take from actor network of TD3
class NCS_net(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(NCS_net, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action		

	def forward(self, state):
		# pdb.set_trace()
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		network_output = torch.sigmoid(self.l3(a))
		return self.max_action * network_output

class GraspValid_net(nn.Module):
	def __init__(self, state_dim):
		super(GraspValid_net, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state):
		# pdb.set_trace()

		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a =	torch.sigmoid(self.l3(a))
		return a

