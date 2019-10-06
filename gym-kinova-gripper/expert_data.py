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

def generate_lifting_data(env, total_steps, filename, grasp_filename):
	states = []
	label = []
	file = open(filename + ".pkl", "rb")
	data = pickle.load(file)
	file.close()

	for step in range(int(total_steps)):
		env.reset()
		curr_states = data["states"][step]	
		obs = env.env.grasp_classifier_reset(curr_states)
		# pdb.set_trace()		
		for _ in range(50):
			# lift
			action = np.array([0.2, 0.0,0.0, 0.0])
			obs, reward , _, _ = env.step(action)

		# record fail or success
		label.append(reward)
		if step % 10000 == 9999:
			print("Steps:{}".format(step + 1))

	print("Finish collecting grasp validation data, saving...")
	grasp_data = {}
	grasp_data["grasp_sucess"] = label
	grasp_data["states"] = data["states"][:]
	grasp_file = open(grasp_filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
	pickle.dump(data, grasp_file)
	grasp_file.close()


# function that generates expert data for grasping an object from ground up
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

	