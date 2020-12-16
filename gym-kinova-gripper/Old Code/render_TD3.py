#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Learn finger control strategy 
# Summer 2019

###############


import os, sys
import numpy as np
import gym
import TD3
import argparse
import torch
import pdb
from main import eval_policy

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="TD3")					# Policy name
	parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0")			# OpenAI gym environment name
	parser.add_argument("--seed", default=2, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e3, type=int)		# Max time steps to run environment for
	parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=128, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.995, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.0005, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	parser.add_argument("--tensorboardlogindex", default=0, type=int)	# tensorboard log index
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	env = gym.make('gym_kinova_gripper:kinovagripper-v0')
	env.seed(args.seed + 100)

	model = "data_cube_5_trained_model_10_07_19_1749.pt"

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = env.action_space.high
	kwargs = {
		"state_dim": state_dim, 
		"action_dim": action_dim, 
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau, 
		"trained_model": model
	}

	kwargs["policy_noise"] = args.policy_noise * max_action
	kwargs["noise_clip"] = args.noise_clip * max_action
	kwargs["policy_freq"] = args.policy_freq
	policy = TD3.TD3(**kwargs)
	# model.eval()
	# policy.actor.load_state_dict(model)
	# obs, done = env.reset(), False
	# reward_total = 0
	# while not done:
	# 	action = policy.select_action(np.array(obs))
	# 	# print(policy.select_action(np.array(obs)))
	# 	obs, reward, done, _ = env.step(action)
	# 	reward_total += reward
	# 	# print(reward_total)
	# 	print(action)
	eval_policy(policy, args.env_name, args.seed, 10)
