#!/usr/bin/env python3

'''
Author : Yi Herng Ong
Date : 9/29/2019
Purpose : Supervised learning for near contact grasping strategy
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
import NCS_nn
# import expert_data
import random
import pandas 
from ounoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def test(env, trained_model):
	actor_net = NCS_nn.NCS_net(48, 4, 0.8).to(device)
	model = torch.load(trained_model)
	actor_net.load_state_dict(model)
	actor_net.eval()

	# IF YOU WAN TO START AT RANDOM INTERMEDIATE STATE
	# file_name = open("data_cube_5_10_07_19_1612.pkl", "rb")
	# data = pickle.load(file_name)
	# states = np.array(data["states"])
	# random_states_index = np.random.randint(0, len(states), size = len(states))

	noise = OUNoise(4)
	expl_noise = OUNoise(4, sigma=0.001)
	for _ in range(10):
		# inference
		obs, done = env.reset(), False
		# obs = env.env.intermediate_state_reset(states[np.random.choice(random_states_index, 1)[0]])
		print("start")
		# while not done:
		for _ in range(150):
			obs = torch.FloatTensor(np.array(obs).reshape(1,-1)).to(device) # + expl_noise.noise()
			action = actor_net(obs).cpu().data.numpy().flatten()
			print(action)
			obs, reward, done, _ = env.step(action)
			# print(reward)

# def train_network(data_filename, max_action, num_epoch, total_steps, batch_size, model_path="trained_model"):
def train_network(training_set, training_label, num_epoch, total_steps, batch_size, model_path="trained_model"):
	# import data
	# file = open(data_filename + ".pkl", "rb")
	# data = pickle.load(file)
	# file.close()

	##### Training Action Net ######
	# state_input = data["states"]
	# actions = data["label"]
	# actor_net = NCS_nn.NCS_net(len(state_input[0]), len(actions[0]), max_action).to(device)
	# criterion = nn.MSELoss()
	# optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)

	##### Training Grasp Classifier ######
	# state_input = data[:, 0:-1]
	# state_input = data["states"]
	# actions = data["grasp_success"]
	# total_steps = data["total_steps"]
	# actions = data[:, -1]
	# pdb.set_trace()
	actor_net = NCS_nn.GraspValid_net(len(state_input[0])).to(device)
	criterion = nn.BCELoss()
	optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)
 
	num_update = total_steps / batch_size

	# print(num_update, len(state_input[0]), len(actions))

	for epoch in range(num_epoch):
			# actions_all_loc = np.random.randint(0,total_steps, size=total_steps)
			# np.random.shuffle(actions_all_loc)
			# actions_all_loc = np.array(actions_all_loc)
			running_loss = 0.0
			start_batch = 0
			end_batch = batch_size

			for i in range(int(num_update)):
				# zero parameter gradients
				optimizer.zero_grad()
				# forward, backward, and optimize
				ind = np.arange(start_batch, end_batch)
				start_batch += batch_size
				end_batch += batch_size
				# states = torch.FloatTensor(np.array(state_input)[ind]).to(device)
				# labels = torch.FloatTensor(np.array(actions)[ind].reshape(-1, 1)).to(device)

				states = torch.FloatTensor(np.array(training_set)[ind]).to(device)
				labels = torch.FloatTensor(np.array(training_label)[ind].reshape(-1, 1)).to(device)				
				# labels = torch.FloatTensor(np.array(actions)[ind]).to(device)

				output = actor_net(states)
				# pdb.set_trace()

				loss = criterion(output, labels)
				loss.backward()
				optimizer.step()
				running_loss += loss.item() 
				# print("loss", loss.item())
				if (i % 100) == 99:
					print("Epoch {} , idx {}, loss: {}".format(epoch + 1, i + 1, running_loss/(100)))
					running_loss = 0.0

	print("Finish training, saving...")
	torch.save(actor_net.state_dict(), model_path + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pt")	
	return actor_net


if __name__ == '__main__':

	# parser = argparse.ArgumentParser()
	# parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0")	# OpenAI gym environment name

	# parser.add_argument("--num_episode", default=1e4, type=int)							# Sets Gym, PyTorch and Numpy seeds
	# parser.add_argument("--batch_size", default=250, type=int)							# Batch size for updating network
	# parser.add_argument("--epoch", default=20, type=int)									# number of epoch

	# parser.add_argument("--data_gen", default=0, type=int)								# bool for whether or not to generate data
	# parser.add_argument("--data", default="data" )										# filename of dataset (the entire traj)
	# parser.add_argument("--grasp_success_data", default="grasp_success_data" )			# filename of grasp success dataset	
	# parser.add_argument("--train", default=1, type=int)									# bool for whether or not to train data
	# parser.add_argument("--model", default="model" )									# filename of model for training	
	# parser.add_argument("--trained_model", default="trained_model" )					# filename of saved model for testing
	# parser.add_argument("--collect_grasp", default=0, type=int )						# check to collect either lift data or grasp data
	# parser.add_argument("--grasp_total_steps", default=100000, type=int )				# number of steps that need to collect grasp data

	# # dataset_cube_2_grasp_10_04_19_1237
	# args = parser.parse_args()

	# if args.data_gen:
	# 	env = gym.make(args.env_name)
	# 	state_dim = env.observation_space.shape[0]
	# 	action_dim = env.action_space.shape[0] 
	# 	max_action = env.action_space.high # action needs to be symmetric
	# 	max_steps = 100 # changes with env
	# 	if args.collect_grasp == 0:
	# 		data = expert_data.generate_Data(env, args.num_episode, args.data)
	# 	else:
	# 		# print("Here")
	# 		data = expert_data.generate_lifting_data(env, args.grasp_total_steps, args.data, args.grasp_success_data)
	# else:

	# 	if args.train:
	# 		assert os.path.exists(args.data + ".pkl"), "Dataset file does not exist"
	# 		# actor_net = train_network(args.data, actor_net, args.epoch, args.num_episode*max_steps, args.batch_size, args.model)
	# 		actor_net = train_network(args.data, 0.8, args.epoch, 400000, args.batch_size, args.model)
		
	# 	else:
	# 		# assert os.path.exists(args.data + ".pkl"), "Dataset file does not exist"
	# 		env = gym.make(args.env_name)
	# 		test(env, args.trained_model)

	# data_filename = "/home/graspinglab/NCS_data/expertdata_01_02_20_1206"
	# data_filename = "/home/graspinglab/NCS_data/Data_Box_S_01_03_20_2309"	
	# data_filename = "/home/graspinglab/NCS_data/Data_Box_M_01_05_20_1705"	
	# data_filename = "/home/graspinglab/NCS_data/Data_Box_B_01_06_20_1532"	
	# data_filename = "/home/graspinglab/NCS_data/Data_Cylinder_S_01_04_20_1701"	
	# data_filename = "/home/graspinglab/NCS_data/Data_Cylinder_M_01_06_20_0013"	
	# data_filename = "/home/graspinglab/NCS_data/Data_Cylinder_B_01_06_20_1922"	

	# num_epoch = 20
	# total_steps = 10000 # not used in the function
	# batch_size = 250
	# model_path = "/home/graspinglab/NCS_data/ExpertTrainedNet"
	# train_network(data_filename, 0.3, 5, total_steps, batch_size, model_path)

	num_epoch = 10
	batch_size = 250
	files_dir = "/home/graspinglab/NCS_data/"
	files = [files_dir + "Data_Box_B_01_06_20_1532", files_dir + "Data_Box_M_01_05_20_1705", files_dir + "Data_Box_S_01_03_20_2309", files_dir + "Data_Cylinder_B_01_06_20_1922",  files_dir + "Data_Cylinder_M_01_06_20_0013", "/home/graspinglab/NCS_data/Data_Cylinder_S_01_04_20_1701"]

	all_training_set = []
	all_training_label = []
	all_testing_set = []
	all_testing_label = []

	for i in range(6):
		file = open(files[i] + ".pkl", "rb")
		data = pickle.load(file)
		file.close()
		state_input = np.array(data["states"])
		grasp_label = np.array(data["grasp_success"])
		
		# extract training set
		training_len = len(state_input) * 0.8
		training_set = state_input[0:int(training_len)]
		training_label = grasp_label[0:int(training_len)]

		# extract testing set
		testing_set = state_input[int(training_len):]
		testing_label = grasp_label[int(training_len):]

		all_training_set = all_training_set + list(training_set)
		all_testing_set = all_testing_set + list(testing_set)
		all_training_label = all_training_label + list(training_label)
		all_testing_label = all_testing_label + list(testing_label)
		print("label: ", i, len(training_set), len(training_label), len(testing_set), len(testing_label))
	# pdb.set_trace()
	train_network(all_training_set, all_training_label, num_epoch, len(all_training_set), batch_size)