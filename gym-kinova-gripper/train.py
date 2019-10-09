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
import expert_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def test(actor_net, trained_model):
	model = torch.load(trained_model)
	actor_net.load_state_dict(model)
	actor_net.eval()
	obs, done = env.reset(), False
	for _ in range(100):
		# inference
		obs = torch.FloatTensor(np.array(obs).reshape(1,-1)).to(device)
		action = actor_net(obs).cpu().data.numpy().flatten()
		print(action)
		obs, reward, done, _ = env.step(action)

def train_network(data_filename, actor_net, num_epoch, total_steps, batch_size, model_path="trained_model"):
	# import data
	file = open(data_filename + ".pkl", "rb")
	data = pickle.load(file)
	file.close()
	state_input = data["states"]
	actions = data["grasp_sucess"]

	# define loss and optimizer
	# criterion = nn.MSELoss()
	criterion = nn.BCELoss()
	optimizer = optim.Adam(actor_net.parameters(), lr=1e-4)

	# number of updates
	num_update = total_steps / batch_size

	for epoch in range(num_epoch):
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
				states = torch.FloatTensor(np.array(state_input)[ind]).to(device)
				labels = torch.FloatTensor(np.array(actions)[ind].reshape(-1, 1)).to(device)
				output = actor_net(states)
				loss = criterion(output, labels)
				# pdb.set_trace()
				loss.backward()
				optimizer.step()
				running_loss += loss.item()
				if i % 100 == 99:
					print("Epoch {} , idx {}, loss: {}".format(epoch + 1, i + 1, running_loss / 100))
					running_loss = 0.0

	print("Finish training, saving...")
	torch.save(actor_net.state_dict(), model_path + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pt")	
	return actor_net


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0")	# OpenAI gym environment name

	parser.add_argument("--num_episode", default=1e4, type=int)							# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--batch_size", default=250, type=int)							# Batch size for updating network
	parser.add_argument("--epoch", default=1, type=int)									# number of epoch

	parser.add_argument("--data_gen", default=0, type=int)								# bool for whether or not to generate data
	parser.add_argument("--data", default="data" )										# filename of dataset (the entire traj)
	parser.add_argument("--grasp_success_data", default="grasp_success_data" )			# filename of grasp success dataset	
	parser.add_argument("--train", default=1, type=int)									# bool for whether or not to train data
	parser.add_argument("--model", default="model" )									# filename of model for training	
	parser.add_argument("--trained_model", default="trained_model" )					# filename of saved model for testing
	parser.add_argument("--collect_grasp", default=0, type=int )						# check to collect either lift data or grasp data
	parser.add_argument("--grasp_total_steps", default=100000, type=int )				# number of steps that need to collect grasp data

	# dataset_cube_2_grasp_10_04_19_1237
	args = parser.parse_args()

	env = gym.make(args.env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = env.action_space.high # action needs to be symmetric
	max_steps = 100 # changes with env
	# actor_net = TD3.Actor(state_dim, action_dim, max_action).to(device)

	if args.data_gen:
		if args.collect_grasp == 0:
			data = expert_data.generate_Data(env, args.num_episode, args.data)
		else:
			# print("Here")
			data = expert_data.generate_lifting_data(env, args.grasp_total_steps, args.data, args.grasp_success_data)
	else:
		actor_net = NCS_nn.NCS_net(state_dim, action_dim, max_action).to(device)
		# actor_net = NCS_nn.GraspValid_net(state_dim).to(device)

		if args.train:
			assert os.path.exists(args.data + ".pkl"), "Dataset file does not exist"
			actor_net = train_network(args.data, actor_net, args.epoch, args.num_episode*max_steps, args.batch_size, args.model)
		else:
			# assert os.path.exists(args.data + ".pkl"), "Dataset file does not exist"
			test(actor_net, args.trained_model)

	 
