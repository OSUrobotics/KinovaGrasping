import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
import pdb
from tensorboardX import SummaryWriter
from ounoise import OUNoise
import pickle
# from pretrain import Pretrain
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	# print("been here", eval_env.seed(seed + 100))

	avg_reward = 0.
	# step = 0
	for i in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			# print("eval act", action)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			# print(avg_reward)
			eval_env.env.render()
	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="TD3")					# Policy name
	parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0")			# OpenAI gym environment name
	parser.add_argument("--seed", default=2, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=3e4, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)		# Max time steps to run environment for
	parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.995, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.0005, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.01, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.05, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	parser.add_argument("--tensorboardindex", default=0, type=int)	# tensorboard log index
	parser.add_argument("--model", default=1, type=int)	# save model index
	parser.add_argument("--pretrained", default=0, type=int)	# tensorboard log index
	
	args = parser.parse_args()

	file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
	print("---------------------------------------")
	print(f"Settings: {file_name}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(args.env_name)
	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	print(action_dim)
	max_action = float(env.action_space.high[0])
	max_action_trained = env.action_space.high # a vector of max actions


	kwargs = {
		"state_dim": state_dim, 
		"action_dim": action_dim, 
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# "trained_model": "data_cube_5_trained_model_10_07_19_1749.pt"		
	}

	# Initialize policy
	if args.policy_name == "TD3": 
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy_name == "OurDDPG": 
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy_name == "DDPG": 		
		policy = DDPG.DDPG(**kwargs)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	

	# Add expert data into replay buffer
	from expert_data import generate_Data
	replay_buffer = generate_Data(env, 300, "random", replay_buffer)


	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env_name, args.seed)] 


	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	model_save_path = "kinova_gripper_learning{}.pt".format(args.model)


	noise = OUNoise(4)
	noise.reset()
	expl_noise = OUNoise(4, sigma=0.001)
	expl_noise.reset()
	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			# action = env.action_space.sample()
			# action = noise.noise().clip(-max_action, max_action)
			obs = torch.FloatTensor(np.array(state).reshape(1,-1)).to(device) 
			action = pretrained_network(obs).cpu().data.numpy().flatten()
			
		else:
			# action = (
			# 	policy.select_action(np.array(state))
			# 	+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			# ).clip(-max_action, max_action)
			# print("action", action)
			action = (policy.select_action(np.array(state)) + expl_noise.noise()).clip(-max_action, max_action)
			# print("training action", action)
		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		# print(action)
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# writer.add_scalar("Episode_reward", episode_reward, t)
		# print("run...")
		# Train agent after collecting sufficient data
		if t % args.batch_size == 99:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")			
			# Reset environment
			state, done = env.reset(), False
			noise.reset()
			expl_noise.reset()
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env_name, args.seed))
			np.save("./results/%s" % (file_name), evaluations)
			print()

	policy.save_model(model_save_path)