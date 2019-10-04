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
# import NCS_nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class expert_PID():
	def __init__(self, action_space):
		self.kp = 1
		self.kd = 1
		self.ki = 1
		self.prev_err = 0.0
		self.sampling_time = 0.04
		self.action_range = [action_space.low, action_space.high]
		self.x = 0.0
		# print(action_space.high)
	# expert vel when object moves
	def get_PID_vel(self, dot_prod):
		err = 1 - dot_prod 
		diff = (err) / self.sampling_time
		vel = err * self.kp + diff * self.kd
		self.prev_err = err
		return vel

	def map_action(self, vel):
		return self.action_range[1][-1]*(vel / 13.637)

	def get_expert_vel(self, dot_prod):
		vel = self.get_PID_vel(dot_prod)
		clipped_vel = self.map_action(vel)
		# pdb.set_trace()
		wrist = 0.0
		f1 = clipped_vel
		f2 = clipped_vel * 0.8
		f3 = clipped_vel * 0.8
		return np.array([wrist, f1, f2, f3])

	def get_expert_move_to_touch(self, dot_prod):
		max_move_vel = self.get_PID_vel(dot_prod)
		clipped_vel = self.map_action(max_move_vel)
		self.x += (self.action_range[1][-1] - clipped_vel) * 0.04
		vel = 0.8 - self.x
		if vel <= 2 * clipped_vel:
			vel = 2 * clipped_vel 
		# pdb.set_trace()
		f1 = vel
		f2 = vel * 0.8
		f3 = vel * 0.8	
		return np.array([0.0, f1, f2, f3])

# function that generates expert data
def generate_Data(env, num_episode, filename):
	# preaction
	wrist = 0.0
	f1 = 0.8
	f2 = f1 * 0.8
	f3 = f1 * 0.8

	# postaction
	wrist_post = 0.0
	f1_post = 0.65
	f2_post = 0.55
	f3_post = 0.55
	
	#lift
	wrist_lift = 0.2
	f1_lift = 0.5
	f2_lift = 0.25
	f3_lift = 0.25

	move_to_touch_action = np.array([wrist, f1, f2, f3])
	move_to_close_action = np.array([wrist_post, f1_post, f2_post, f3_post])
	move_to_lift = np.array([wrist_lift, f1_lift, f2_lift, f3_lift])

	expert = expert_PID(env.action_space)
	episode_timesteps = 0

	# open file
	file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')

	label = []
	states = []
	for episode in range(num_episode):

		obs, done = env.reset(), False
		ini_dot_prod = obs[-5]
		action = expert.get_expert_move_to_touch(ini_dot_prod)
		touch = 0
		for t in range(100):
			# print(action)
			states.append(obs)
			label.append(action)
			# print(len(obs))
			next_obs, reward, done, _ = env.step(action)
			# done_bool = float(done) if t < env._max_episode_steps else 0
			# replay_buffer.add(obs, action, next_obs, reward, done_bool)
			obs = next_obs
			dot_prod = obs[-5]
			# print(dot_prod)
			action = expert.get_expert_move_to_touch(dot_prod)
			if abs(dot_prod - ini_dot_prod) > 0.001:
				action = expert.get_expert_vel(dot_prod)
				touch = 1
				# print(action)
			if dot_prod > 0.8 and touch == 1:
				action = move_to_close_action
			if t > 60 and touch == 1:
				action = move_to_lift

		print("Collecting.., num_episode:{}".format(episode))
	print("saving...")
	data = {}
	data["states"] = states
	data["label"] = label
	pickle.dump(data, file)
	file.close()

	return data

def test(actor_net, trained_model):
	model = torch.load(trained_model)
	actor_net.load_state_dict(model)
	actor_net.eval()
	obs, done = env.reset(), False
	for _ in range(100):
		# inference
		obs = torch.FloatTensor(np.array(obs).reshape(1,-1)).to(device)
		action = actor_net(obs).cpu().data.numpy().flatten()
		# print(action)
		obs, reward, done, _ = env.step(action)

def train_network(data_filename, actor_net, num_epoch, total_steps, batch_size, model_path="trained_model"):
	# import data
	file = open(data_filename + ".pkl", "rb")
	data = pickle.load(file)
	file.close()
	state_input = data["states"]
	actions = data["label"]

	# define loss and optimizer
	criterion = nn.MSELoss()
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
				labels = torch.FloatTensor(np.array(actions)[ind]).to(device)
				output = actor_net(states)
				loss = criterion(output, labels)
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
	parser.add_argument("--batch_size", default=100, type=int)							# Batch size for updating network
	parser.add_argument("--epoch", default=1, type=int)									# number of epoch
	parser.add_argument("--data_gen", default=0, type=int)								# bool for generating data
	parser.add_argument("--filename", default="data" )									# filename of dataset
	parser.add_argument("--save_model", default=1, type=int)							# save model
	parser.add_argument("--trained_model", default="trained_model" )					# filename of saved model

	
	args = parser.parse_args()

	env = gym.make(args.env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = env.action_space.high # action needs to be symmetric
	max_steps = 100 # changes with env
	actor_net = TD3.Actor(state_dim, action_dim, max_action).to(device)
	if args.data_gen:
		data = generate_Data(env, args.num_episode, args.filename)
	else:
		assert os.path.exists(args.filename + ".pkl"), "Dataset file does not exist"
		actor_net = train_network(args.filename, actor_net, args.epoch, args.num_episode*max_steps, args.batch_size, args.trained_model)


# obs[36:44] 
