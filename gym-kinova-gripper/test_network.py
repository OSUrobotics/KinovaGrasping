#!/usr/bin/env python

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from train import train_network
import csv
import pandas as pd
import numpy as np
import pdb
import NCS_nn
import random
import pickle
import datetime
from ounoise import OUNoise
import gym
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraspValid_net(nn.Module):
	def __init__(self, state_dim):
		super(GraspValid_net, self).__init__()
		self.l1 = nn.Linear(state_dim, 10)
		self.l2 = nn.Linear(10, 1)
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a =	torch.sigmoid(self.l2(a))
		# pdb.set_trace()
		return a


def label_grasp_data():
	data_filename = "data_cube_9_10_17_19_1704"
	file = open(data_filename + ".pkl", "rb")
	data = pickle.load(file)
	file.close()	
	grasp_success = np.array(data["states_ready_grasp"])
	actions_success = np.array(data["label_ready_grasp"])
	close_grasp = np.array(data["states_when_closing"])
	close_action = np.array(data["label_when_closing"])


	new_close_grasp = []
	for i in range(len(close_grasp)):
		temp = np.append(close_grasp[i], 0)
		new_close_grasp.append(temp)	
		# pdb.set_trace()

	new_grasp_success = []
	for j in range(len(grasp_success)):
		temp1 = np.append(grasp_success[j], 1)
		new_grasp_success.append(temp1)


	all_samples = np.concatenate((np.array(new_close_grasp), np.array(new_grasp_success)))
	np.random.shuffle(all_samples)
	np.random.shuffle(all_samples)
	newfile = open("data_cube_9_grasp" + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
	pickle.dump(all_samples, newfile)
	newfile.close()

	# all_samples = np.array(all_samples)
	# print(len(np.where(all_samples[:, -1]==1.0)[0]))
	# print(len(new_close_grasp))

def test_network_structure():
	x = np.array([1,0,0,1,0,0,1,1,0,0,0,1,0,1])
	y = np.array([0,1,1,0,1,1,0,0,1,1,1,0,1,0])

	x = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0])
	y = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1])

	actor_net = NCS_nn.GraspValid_net(1).to(device)
	criterion = nn.BCELoss()
	optimizer = optim.SGD(actor_net.parameters(), lr=5e-1)
	updates = 14 / 2
	for _ in range(10):
		s = 0
		end = 2
		running_loss = 0.0
		for i in range(int(updates)):
			r = np.arange(s, end)
			inputs = torch.FloatTensor(x[r].reshape(-1, 1)).to(device)
			labels = torch.FloatTensor(y[r].reshape(-1, 1)).to(device)
			s += 2
			end += 2
			optimizer.zero_grad()
			outputs = actor_net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 2 == 1:
				print(running_loss / 2)
				running_loss = 0.0


def test_grasp_classifier():
	data_filename = "/home/graspinglab/NCS_data/data_cube_9_grasp_10_17_19_1709"
	file = open(data_filename + ".pkl", "rb")
	data = pickle.load(file)
	file.close()	
	state_input = data[400000:500000, 0:-1]
	label = data[400000:500000, -1]

	grasp_net = NCS_nn.GraspValid_net(48).to(device)
	trained_model = "/home/graspinglab/NCS_data/data_cube_9_grasp_classifier_10_17_19_1734.pt"
	model = torch.load(trained_model)
	grasp_net.load_state_dict(model)
	grasp_net.eval()

	def test_random_action():
		env = gym.make('gym_kinova_gripper:kinovagripper-v0')
		obs, done = env.reset(), False
		noise = OUNoise(3)
		max_action = float(env.action_space.high[0])
		correct = 0
		noise.reset()
		cum_reward = 0.0
		for i in range(100):
			finger_actions = noise.noise().clip(-max_action, max_action)
			# actions = np.array([0.0, finger_actions[0], finger_actions[1], finger_actions[2]])
			actions = np.array([0.4, 0.5, 0.5, 0.5])
			obs, reward, done, _ = env.step(actions)
			inputs = torch.FloatTensor(np.array(obs)).to(device)
			# print(obs[41:47]) # finger obj distance
			# print(obs[35:41]) # finger obj distance
			# cum_reward += reward
			# print(cum_reward)
			# print(np.max(np.array(obs[35:38]) - 0.0175), np.max(np.array(obs[38:41]) - 0.0175))
			# if obs[20] < 0.07:

			# Condition where we can use the grasp classifier
			# if (np.max(np.array(obs[41:47])) - 0.035) < 0.01 or (np.max(np.array(obs[35:41])) - 0.015) < 0.01: 
			# 	outputs = grasp_net(inputs).cpu().data.numpy().flatten()
				# print(outputs)
				# if outputs == 1.0:
					# pdb.set_trace()

	def test_expert_action(state_input, actions):
		correct = 0.0
		for i in range(100000):
			inputs = torch.FloatTensor(state_input[i]).to(device)
			outputs = grasp_net(inputs).cpu().data.numpy().flatten()
			if outputs > 0.5 and actions[i] == 1:
				correct += 1
			if outputs <= 0.5 and actions[i] == 0:
				correct += 1	
			# pdb.set_trace()
			# if outputs == y[i]:
			# 	correct += 1
			# correct += (predicted == test_y)
		print("Accuracy", correct / 100000)

	# test_expert_action(state_input, label)
	test_random_action()

def extract_jA_4_action():
	data_filename = "/home/graspinglab/NCSGen/gym-kinova-gripper/test_jA_12_23_19_1604"
	file = open(data_filename + ".pkl", "rb")
	data = pickle.load(file)
	file.close()	
	states = np.array(data["states_all_episode"])
	jA_all_episode = []

	for i in range(10): # episode
		# states[i]
		episode = states[i][:]
		jA_arr = []
		for j in range(100): # states
			jA = episode[j, 24:31]
			jA_arr.append(jA)
		jA_all_episode.append(jA_arr)
	# pdb.set_trace()

	file_save = open("jA_action" + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
	pickle.dump(jA_all_episode, file_save)
	file_save.close()

def test_jA_actions():
	data_filename = "/home/graspinglab/NCSGen/gym-kinova-gripper/jA_action_12_23_19_1606"
	file = open(data_filename + ".pkl", "rb")
	data = pickle.load(file)
	file.close()	
	data = np.array(data)
	# states = np.array(data["states_all_episode"])	
	env = gym.make('gym_kinova_gripper:kinovagripper-v0')
	# pdb.set_trace()

	for i in range(10):
		episode = data[i][:]
		env.reset()
		# env.env.reset([0.0, 0.0])
		for j in range(100):
			actions = episode[j, 0:4]
			obs, reward, done, _ = env.step(actions)
			env.render()
			# print(actions)

		pdb.set_trace()

if __name__ == '__main__':
	# test_grasp_classifier()
	# label_grasp_data()
	# extract_jA_4_action()
	test_jA_actions()