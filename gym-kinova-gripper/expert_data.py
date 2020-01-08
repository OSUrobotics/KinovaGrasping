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
from copy import deepcopy
# from gen_new_env import gen_new_obj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################################################################

###########################################
# Previous PID Nudge Controller w/ 0.8 rad/s joint velocity
###########################################
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
	env = gym.make('gym_kinova_gripper:kinovagripper-v0')
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
	states_all_episode = []
	obs_all_episode = []
	action_all_episode = []
	nextobs_all_episode = []
	reward_all_episode = []
	done_all_episode = []

	for episode in range(num_episode):
		# env = gym.make('gym_kinova_gripper:kinovagripper-v0')
		obs, done = env.reset(), False
		dom_finger = env.env._get_obj_pose()[0] # obj's position in x
		ini_dot_prod = env.env._get_dot_product(env.env._get_obj_pose()) # 
		action = expert.get_expert_move_to_touch(ini_dot_prod, dom_finger)
		touch = 0
		not_close = 1
		close = 0
		touch_dot_prod = 0.0
		t = 0	
		cum_reward = 0.0
		states_each_episode = []
		obs_each_episode = []
		action_each_episode = []
		nextobs_each_episode = []
		reward_each_episode = []
		done_each_episode = []

		for _ in range(100):
			states.append(obs)
			label.append(action)	
			states_each_episode.append(obs)
			
			obs_each_episode.append(obs)

			# pdb.set_trace()
			next_obs, reward, done, _ = env.step(action)
			jA_action = next_obs[24:28][:]
			jA_action[0] = (jA_action[0] / 0.2) * 1.5
			action_each_episode.append(jA_action) # get joint angle as action
			nextobs_each_episode.append(next_obs)
			reward_each_episode.append(reward)
			done_each_episode.append(done)

			# env.render()
			# print(action)
			# store data into replay buffer 
			replay_buffer.add(obs, action, next_obs, reward, done)
			# replay_buffer.add(obs, obs[24:28], next_obs, reward, done) # store joint angles as actions

			cum_reward += reward
			# print(next_obs[0:7])
			obs = next_obs
			# dot_prod = obs[-1]
			dot_prod = env.env._get_dot_product(env.env._get_obj_pose())			
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
			# print(next_obs[24:31])
		# pdb.set_trace()
		states_all_episode.append(states_each_episode)
		obs_all_episode.append(obs_each_episode)
		action_all_episode.append(action_each_episode)
		nextobs_all_episode.append(nextobs_each_episode)
		reward_all_episode.append(reward_each_episode)
		done_all_episode.append(done_each_episode)

		# print("Collecting.., num_episode:{}".format(episode))
		# pdb.set_trace()
	# print("saving...")
	# data = {}
	# data["states_all_episode"] = states_all_episode
	# pdb.set_trace()

	# data["states"] = states
	# data["label"] = label
	# data["states_for_lifting"] = states_for_lifting
	# data["label_for_lifting"] = label_for_lifting
	# data["states_when_closing"] = states_when_closing
	# data["label_when_closing"] = label_when_closing	
	# data["states_ready_grasp"] = states_ready_grasp
	# data["label_ready_grasp"] = label_ready_grasp

	### Data collection for joint angle action space ###
	# data["obs"] = obs_all_episode
	# data["action"] = action_all_episode
	# data["next_obs"] = nextobs_all_episode
	# data["reward"] = reward_all_episode
	# data["done"] = done_all_episode
	# file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
	# pickle.dump(data, file)
	# file.close()
	# return data

	return replay_buffer

#################################################################################################



class PID(object):
	def __init__(self, action_space):
		self.kp = 1
		self.kd = 1
		self.ki = 1
		self.prev_err = 0.0
		self.sampling_time = 4
		self.action_range = [action_space.low, action_space.high]

	def velocity(self, dot_prod):
		err = 1 - dot_prod
		diff = (err) / self.sampling_time
		vel = err * self.kp + diff * self.kd
		action = (vel / 1.25) * 0.3 # 1.25 means dot product equals to 1
		if action < 0.05:
			action = 0.05
		return action

	def joint(self, dot_prod):
		err = 1 - dot_prod 
		diff = (err) / self.sampling_time
		joint = err * self.kp + diff * self.kd
		action = (joint / 1.25) * 2 # 1.25 means dot product equals to 1
		return action

	def touch_vel(self, obj_dotprod, finger_dotprod):
		err = obj_dotprod - finger_dotprod
		diff = err / self.sampling_time
		vel = err * self.kp + diff * self.kd
		action = (vel) * 0.3
		if action < 0.05:
			action = 0.05
		return action

##############################################################################
### PID nudge controller ###
# 1. Obtain (noisy) initial position of an object
# 2. move fingers that closer to the object
# 3. Move the other when the object is almost at the center of the hand 
# 4. Close grasp

### PID nudge controller ###
# 1. Obtain (noisy) initial position of an object
# 2. Move fingers that further away to the object 
# 3. Close the other finger (nearer one) and make contact "simultaneously"
# 4. Close fingers to secure grasp
##############################################################################
class ExpertPIDController(object):
	def __init__(self, states):
		self.prev_f1jA = 0.0
		self.prev_f2jA = 0.0
		self.prev_f3jA = 0.0
		self.step = 0.0
		self.init_obj_pose = states[21]
		self.init_dot_prod = states[-7]

	def _count(self):
		self.step += 1

	def NudgeController(self, states, action_space, label):
		# Define pid controller
		pid = PID(action_space)

		# obtain robot and object state
		robot_pose = np.array([states[0], states[1], states[2]])
		obj_pose = states[21]
		obj_dot_prod = states[-7]
		f1_jA = states[25]
		f2_jA = states[26]
		f3_jA = states[27]
		# Define target region
		x = 0.0
		y = -0.01
		target_region = [x, y]
		max_vel = 0.3
		# Define finger action
		f1 = 0.0
		f2 = 0.0
		f3 = 0.0
		wrist = 0.0

		if abs(self.init_obj_pose) <= 0.03:
			f1, f2, f3 = 0.2, 0.2, 0.2
			if abs(obj_dot_prod - self.init_dot_prod) > 0.01:
				f1, f2, f3 = 0.2, 0.1, 0.1
				if self.step > 200:
					wrist = 0.3	
		else:	
			# object on right hand side, move 2-fingered side
			if self.init_obj_pose < 0.0:
				# Pre-contact
				if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
					f2 = pid.touch_vel(obj_dot_prod, states[-5])
					f3 = f2
					f1 = 0.0
					wrist = 0.0
				# Post-contact	
				else:
					if abs(1 - obj_dot_prod) > 0.01:
						f2 = pid.velocity(obj_dot_prod)
						f3 = f2
						f1 = 0.05
						wrist = 0.0
					else:
						f1 = pid.touch_vel(obj_dot_prod, states[-6])
						f2 = 0.0
						f3 = 0.0
						wrist = 0.0
					# Hand tune lift time
					if self.step > 400:
						f1, f2, f3 = 0.3, 0.15, 0.15
						wrist = 0.3
			# object on left hand side, move 1-fingered side
			else:
				# Pre-contact
				if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
					f1 = pid.touch_vel(obj_dot_prod, states[-6])
					f2 = 0.0
					f3 = 0.0
					wrist = 0.0
				# Post-contact	
				else:
					if abs(1 - obj_dot_prod) > 0.01:
						f1 = pid.velocity(obj_dot_prod)
						f2 = 0.05
						f3 = 0.05
						wrist = 0.0
					else:
						f2 = pid.touch_vel(obj_dot_prod, states[-5])
						f3 = f2
						f1 = 0.0
						wrist = 0.0
					# Hand tune lift time
					if self.step > 400:
						f1, f2, f3 = 0.3, 0.15, 0.15
						wrist = 0.3

		if self.step <= 400:
			label.append(0)
		else:
			label.append(1)
		self._count()
		# print(self.step)

		return np.array([wrist, f1, f2, f3]), label


def GenerateExpertPID_JointVel(episode_num, replay_buffer, save=True):
	env = gym.make('gym_kinova_gripper:kinovagripper-v0')
	# episode_num = 10
	obs_label = []
	grasp_label = []
	action_label = []
	total_steps = 0
	for i in range(episode_num):
		obs, done = env.reset(), False
		controller = ExpertPIDController(obs)
		replay_buffer.add_episode(1)
		while not done:
			obs_label.append(obs)
			action, grasp_label = controller.NudgeController(obs, env.action_space, grasp_label)
			action_label.append(action)
			next_obs, reward, done, _ = env.step(action)
			replay_buffer.add(obs, action, next_obs, reward, float(done))
			obs = next_obs
			total_steps += 1

		replay_buffer.add_episode(0)
		# print(i)
	if save:
		filename = "expertdata"
		print("Saving...")
		data = {}
		data["states"] = obs_label
		data["grasp_success"] = grasp_label
		data["action"] = action_label
		data["total_steps"] = total_steps
		file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
		pickle.dump(data, file)
		file.close()

	return replay_buffer
	
# def GenerateExpertPID_JointAngle():

def store_saved_data_into_replay(replay_buffer, num_episode):
	filename = "/home/graspinglab/NCSGen/gym-kinova-gripper/collect_jA_12_27_19_1458"
	file = open(filename + ".pkl", "rb")
	data = pickle.load(file)
	obs = data["obs"]
	action = data["action"]
	next_obs = data["next_obs"]
	reward = data["reward"]
	done = data["done"]
	file.close()	
	# pdb.set_trace()

	# env = gym.make('gym_kinova_gripper:kinovagripper-v0')


	for i in range(num_episode):
		obs_episode = obs[i][:]
		action_episode = action[i][:]
		next_obs_episode = next_obs[i][:]
		reward_episode = reward[i][:]
		done_episode = done[i][:]
		# env.reset()
		# t_r = 0.0
		for j in range(100):
			# ob, re, do, _ = env.step(action_episode[j])
			# env.render()
			replay_buffer.add(obs_episode[j], action_episode[j], next_obs_episode[j], reward_episode[j], done_episode[j]) # store joint angles as actions
			# t_r += re
			# print(action_episode[j])
		# print(t_r)
		# pdb.set_trace()

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

# testing #
# GenerateExpertPID_JointVel(4000)