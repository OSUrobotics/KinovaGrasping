#!/usr/bin/env python3

'''
Author : Yi Herng Ong
Date : 9/29/2019
Purpose : Supervised learning for near contact grasping strategy
'''

import os, sys
import numpy as np
import gym
import argparse
import pdb
import pickle
import datetime
from NCS_nn import NCS_net, GraspValid_net
import torch 

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
		self.flag = 1
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

	def get_expert_vel(self, dot_prod, dominant_finger):
		vel = self.get_PID_vel(dot_prod)
		clipped_vel = self.map_action(vel)
		# pdb.set_trace()
		wrist = 0.0
		if dominant_finger > 0:
			f1 = clipped_vel
			f2 = clipped_vel * 0.8 
			f3 = clipped_vel * 0.8
		else:
			f1 = clipped_vel * 0.8
			f2 = clipped_vel 
			f3 = clipped_vel 
		return np.array([wrist, f1, f2, f3])

	def get_expert_move_to_touch(self, dot_prod, dominant_finger):
		max_move_vel = self.get_PID_vel(dot_prod)
		clipped_vel = self.map_action(max_move_vel)
		self.x += (self.action_range[1][-1] - clipped_vel) * 0.04
		if self.x >= clipped_vel: # if x accumulate to clipped vel, use clipped vel instead
			self.x = clipped_vel
		if self.x < 0.0:
			self.x = 0.0
		vel = 0.8 - self.x
		# pdb.set_trace()
		# if dominant_finger > 0:
		# 	f1 = vel
		# 	f2 = vel * 0.8
		# 	f3 = vel * 0.8
		# else:
		f1 = vel
		f2 = vel 
		f3 = vel 
		return np.array([0.0, f1, f2, f3])

	def generate_expert_move_to_close(self, vel, max_vel, dominant_finger):
		# if self.flag == 1:
		# 	# print("only here once")
		self.vel = vel[:]
		# self.flag = 0
		wrist = 0.0
		# f_vel = self.vel[:]
		for i in range(3):
			if self.vel[i+1] < max_vel:
				# print("here")
				self.vel[i+1] += 0.1*self.vel[i+1]
			else:
				self.vel[i+1] = max_vel
		return np.array([wrist, self.vel[1], self.vel[2]*0.7, self.vel[3]*0.7])


def generate_lifting_data(env, total_steps, filename, grasp_filename):
	states = []
	label = []
	# file = open(filename + ".pkl", "rb")
	# data = pickle.load(file)
	# file.close()
	import time
	print(total_steps)
	for step in range(int(total_steps)):
		_, curr_actions, reward = env.reset()
		if reward == 1:
			continue
		else:	
			for i in range(40):
				# lift
				if i < 20:
					action = np.array([0.0, 0.0, 0.0, 0.0])
				else:
					action = np.array([0.2, curr_actions[1], curr_actions[2], curr_actions[3]])
				# pdb.set_trace()
				obs, reward , _, _ = env.step(action)
		
			# time.sleep(0.25)	
		print(reward)

		# record fail or success
		label.append(reward)
		if step % 10000 == 9999:
			print("Steps:{}".format(step + 1))

	print("Finish collecting grasp validation data, saving...")
	grasp_data = {}
	grasp_data["grasp_sucess"] = label
	grasp_data["states"] = data["states"][0:int(total_steps)]
	grasp_file = open(grasp_filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
	pickle.dump(grasp_data, grasp_file)
	grasp_file.close()


# function that generates expert data for grasping an object from ground up
def generate_Data(env, num_episode, filename, replay_buffer):
	# grasp_net = GraspValid_net(35).to(device)
	# trained_model = "data_cube_7_grasp_classifier_10_16_19_1509.pt"

	# model = torch.load(trained_model)
	# grasp_net.load_state_dict(model)
	# grasp_net.eval()

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
	f1_lift = 0.4
	f2_lift = 0.4
	f3_lift = 0.4

	move_to_touch_action = np.array([wrist, f1, f2, f3])
	move_to_close_action = np.array([wrist_post, f1_post, f2_post, f3_post])
	move_to_lift = np.array([wrist_lift, f1_lift, f2_lift, f3_lift])

	expert = expert_PID(env.action_space)
	episode_timesteps = 0

	# display filename
	# print("filename:", filename)

	label = []
	states = []
	states_for_lifting = []
	label_for_lifting = []
	states_when_closing = []
	label_when_closing = []
	states_ready_grasp = []
	label_ready_grasp = []
	for episode in range(num_episode):
		obs, done = env.reset(), False
		ini_dot_prod = env.env._get_dot_product() # 
		dom_finger = env.env._get_obj_pose()[0] # obj's position in x
		action = expert.get_expert_move_to_touch(ini_dot_prod, dom_finger)
		touch = 0
		not_close = 1
		close = 0
		touch_dot_prod = 0.0
		t = 0	
		cum_reward = 0.0
		for _ in range(100):
			states.append(obs)
			label.append(action)	
			next_obs, reward, done, _ = env.step(action)
			# env.render()
			# store data into replay buffer 
			replay_buffer.add(obs, action, next_obs, reward, done)
			cum_reward += reward
			# print(next_obs[0:7])
			obs = next_obs
			# dot_prod = obs[-1]
			dot_prod = env.env._get_dot_product()			
			# move closer towards object
			if touch == 0:
				action = expert.get_expert_move_to_touch(dot_prod, dom_finger)
			# contact with object, PID control with object pose as feedback
			if abs(dot_prod - ini_dot_prod) > 0.001 and not_close == 1:				
				action = expert.get_expert_vel(dot_prod, dom_finger)
				prev_vel = action
				touch_dot_prod = dot_prod
			# if object is close to center 
			if touch_dot_prod > 0.8: # can only check dot product after fingers are making contact
				close = 1				
			if close == 1:
				action = expert.generate_expert_move_to_close(prev_vel, 0.6, dom_finger)
				not_close = 0
				# when it's time to lift
				states_for_lifting.append(obs)
				label_for_lifting.append(action)
				if t > 60:
					action[0] = 0.8
			if t > 50:
				states_ready_grasp.append(obs)
				label_ready_grasp.append(action)	
			if t <= 50:
				states_when_closing.append(obs)
				label_when_closing.append(action)
			t += 1
		# print("Collecting.., num_episode:{}".format(episode))
		# pdb.set_trace()
	# print("saving...")
	# data = {}
	# data["states"] = states
	# data["label"] = label
	# data["states_for_lifting"] = states_for_lifting
	# data["label_for_lifting"] = label_for_lifting
	# data["states_when_closing"] = states_when_closing
	# data["label_when_closing"] = label_when_closing	
	# data["states_ready_grasp"] = states_ready_grasp
	# data["label_ready_grasp"] = label_ready_grasp
	# file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
	# pickle.dump(data, file)
	# file.close()
	# return data

	return replay_buffer



# def generate_closing_data(env, num_episode, filename):

# Command line
'''
# Collect entire sequence / trajectory
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so python train.py --num_episode 5000 --data_gen 1 --filename data_cube_5 

# Collect grasp data
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so python train.py --grasp_total_steps 10 --filename data_cube_5_10_07_19_1612 --grasp_filename data_cube_5_10_07_19_1612_grasp --grasp_validation 1 --data_gen 1

# Training
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so python train.py --grasp_validation 1 --filename data_cube_5_10_07_19_1612 --trained_model data_cube_5_trained_model --num_episode 5000
'''