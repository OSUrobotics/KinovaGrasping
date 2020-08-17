import numpy as np
import torch
import gym
import argparse
import os, sys

import utils
import TD3
import OurDDPG
import DDPG
import DDPGfD
import pdb
from tensorboardX import SummaryWriter
from ounoise import OUNoise
import pickle
import datetime
import csv
import time
from expert_data import generate_Data, store_saved_data_into_replay, GenerateExpertPID_JointVel, GenerateTestPID_JointVel

# 'gym_kinova_gripper:kinovagripper-v0'
# from pretrain import Pretrain
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

def save_coordinates(x,y,filename):
	np.save(filename+"_x_arr", x)
	np.save(filename+"_y_arr", y)

# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, requested_shapes, requested_orientation, mode, eval_episodes=200, compare=True):
	num_success=[0,0]
	# Heatmap plot success/fail object coordinates
	seval_obj_posx = np.array([])
	feval_obj_posx = np.array([])
	seval_obj_posy = np.array([])
	feval_obj_posy = np.array([])

	# Folder to save heatmap coordinates
	evplot_saving_dir = "./eval_plots"
	if not os.path.isdir(evplot_saving_dir):
		os.mkdir(evplot_saving_dir)

	if compare:
		eval_env = gym.make(env_name)
		eval_env.seed(seed + 100)

		# Generate randomized list of objects to select from
		eval_env.Generate_Latin_Square(eval_episodes,"eval_objects.csv",shape_keys=requested_shapes)

		avg_reward = 0.0
		# step = 0
		for i in range(40):
			if i<23:
				x=(i)*0.005-0.055
				y=0.0
			elif i>=23:
				x=(i-23)*0.005-0.04
				y=0.0
			print('started pos', i)
			success=0
			#eval_env = gym.make(env_name)
			state, done = eval_env.reset(start_pos=[x,y],env_name="eval_env",shape_keys=requested_shapes,hand_orientation=requested_orientation,mode=mode), False
			cumulative_reward = 0

			# Keep track of object coordinates
			obj_coords = eval_env.get_obj_coords()

			while not done:
				action = policy.select_action(np.array(state[0:48]))
				state, reward, done, _ = eval_env.step(action)
				avg_reward += reward
				cumulative_reward += reward
				if reward > 25:
					success=1
				# eval_env.render()
				# print(reward)
			# pdb.set_trace()
			# print(cumulative_reward)
			num_success[0]+=success
			state, done = eval_env.reset(start_pos=[x,y],env_name="eval_env",shape_keys=requested_shapes,hand_orientation=requested_orientation,mode=mode), False
			success=0
			while not done:
				action = GenerateTestPID_JointVel(np.array(state[0:48]),eval_env)
				state, reward, done, _ = eval_env.step(action)
				avg_reward += reward
				cumulative_reward += reward
				if reward > 25:
					success=1
			num_success[1]+=success

			# Save initial object coordinate as success/failure
			if(success):
				x_val = (obj_coords[0])
				y_val = (obj_coords[1])
				x_val = np.asarray(x_val).reshape(1)
				y_val = np.asarray(y_val).reshape(1)
				seval_obj_posx = np.append(seval_obj_posx,x_val)
				seval_obj_posy = np.append(seval_obj_posy,y_val)
			else:
				x_val = (obj_coords[0])
				y_val = (obj_coords[1])
				x_val = np.asarray(x_val).reshape(1)
				y_val = np.asarray(y_val).reshape(1)
				feval_obj_posx = np.append(feval_obj_posx,x_val)
				feval_obj_posy = np.append(feval_obj_posy,y_val)

		avg_reward /= eval_episodes

		print("---------------------------------------")
		# print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
		print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
		print("---------------------------------------")

	else:
		eval_env = gym.make(env_name)
		eval_env.seed(seed + 100)

		# Generate randomized list of objects to select from
		eval_env.Generate_Latin_Square(eval_episodes,"eval_objects.csv",shape_keys=requested_shapes)

		avg_reward = 0.0
		# step = 0
		for i in range(eval_episodes):
			success=0
			#eval_env = gym.make(env_name)
			state, done = eval_env.reset(hand_orientation=requested_orientation,mode=args.mode,shape_keys=requested_shapes,env_name="eval_env"), False
			cumulative_reward = 0

			while not done:
				action = policy.select_action(np.array(state[0:48]))
				state, reward, done, _ = eval_env.step(action)
				avg_reward += reward
				cumulative_reward += reward
				if reward > 25:
					success=1
				# eval_env.render()
				# print(reward)
			# pdb.set_trace()
			# print(cumulative_reward)
			num_success+=success

			# Save initial object coordinate as success/failure
			if(success):
				x_val = (obj_coords[0])
				y_val = (obj_coords[1])
				x_val = np.asarray(x_val).reshape(1)
				y_val = np.asarray(y_val).reshape(1)
				seval_obj_posx = np.append(seval_obj_posx,x_val)
				seval_obj_posy = np.append(seval_obj_posy,y_val)
			else:
				x_val = (obj_coords[0])
				y_val = (obj_coords[1])
				x_val = np.asarray(x_val).reshape(1)
				y_val = np.asarray(y_val).reshape(1)
				feval_obj_posx = np.append(feval_obj_posx,x_val)
				feval_obj_posy = np.append(feval_obj_posy,y_val)

		avg_reward /= eval_episodes

		print("---------------------------------------")
		# print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
		print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
		print("---------------------------------------")

	ret = [avg_reward, num_success, seval_obj_posx,seval_obj_posy,feval_obj_posx,feval_obj_posy,total_evalx,total_evaly]
	return ret


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="DDPGfD")				# Policy name
	parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0")			# OpenAI gym environment name
	parser.add_argument("--seed", default=2, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=100, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=100, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)		# Max time steps to run environment for
	parser.add_argument("--max_episode", default=2000, type=int)		# Max time steps to run environment for
	parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=250, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.995, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.0005, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.01, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.05, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	parser.add_argument("--tensorboardindex", default="new")	# tensorboard log name
	parser.add_argument("--model", default=1, type=int)	# save model index
	parser.add_argument("--pre_replay_episode", default=100, type=int)	# Number of episode for loading expert trajectories
	parser.add_argument("--saving_dir", default="new")	# Number of episode for loading expert trajectories
	parser.add_argument("--shapes", action='store', type=str) # Requested shapes to use (in format of object keys)
	parser.add_argument("--hand_orientation", action='store', type=str) # Requested shapes to use (in format of object keys)
	parser.add_argument("--mode", action='store', type=str, default="train") # Mode to run experiments with (train, test, etc.)

	args = parser.parse_args()

	file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
	print("---------------------------------------")
	print("Settings: {file_name}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(args.env_name)
	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	#print ("STATE DIM ---------", state_dim)
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])
	max_action_trained = env.action_space.high # a vector of max actions

	# Sets list of desired objects for experiment
	requested_shapes = args.shapes
	requested_shapes = requested_shapes.split(',')

	print("args.shapes",args.shapes)
	print("requested_shapes: ",requested_shapes)

	# Set the desired hand orientation (normal or random)
	requested_orientation = args.hand_orientation

	print("Requested Hand orientation: ", requested_orientation)

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
	elif args.policy_name == "DDPGfD":
		policy = DDPGfD.DDPGfD(**kwargs)


	# Initialize replay buffer with expert demo
	print("----Generating {} expert episodes----".format(args.pre_replay_episode))
	# Not being used (commented out in other files)
	#from expert_data import generate_Data, store_saved_data_into_replay, GenerateExpertPID_JointVel

	# from pretrain_from_RL import pretrain_from_agent
	# expert_policy = DDPGfD.DDPGfD(**kwargs)
	# replay_buffer = pretrain_from_agent(expert_policy, env, replay_buffer, args.pre_replay_episode)

	# trained policy
	#policy.load("./policies/reward_all/DDPGfD_kinovaGrip_10_22_19_2151")

	# **Uncomment***
	policy.load('./policies/exp2s1_wo_graspclassifier/DDPGfD_kinovaGrip_01_14_20_1041')
	#policy.load('/Users/vanil/5_3_20_Heatmap/NCSGen_obj_pos_plot/gym-kinova-gripper/policies/exp2s1_wo_graspclassifier/DDPGfD_kinovaGrip_01_14_20_1041')

	# old pid control
	replay_buffer = utils.ReplayBuffer_episode(state_dim, action_dim, env._max_episode_steps, args.pre_replay_episode, args.max_episode)
	# replay_buffer = generate_Data(env, args.pre_replay_episode, "random", replay_buffer)
	# replay_buffer = store_saved_data_into_replay(replay_buffer, args.pre_replay_episode)

	# new pid control
	# replay_buffer = utils.ReplayBuffer_VarStepsEpisode(state_dim, action_dim, args.pre_replay_episode)
	# replay_buffer = GenerateExpertPID_JointVel(args.pre_replay_episode, replay_buffer, False)

	# Evaluate untrained policy
	evaluations = []

	# Fill pre-training object list using latin square
	env.Generate_Latin_Square(args.max_episode,"objects.csv",shape_keys=requested_shapes)

	state, done = env.reset(env_name="env",shape_keys=requested_shapes,hand_orientation=requested_orientation,mode=args.mode), False

	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# Check and create directory
	saving_dir = "./policies/" + args.saving_dir
	if not os.path.isdir(saving_dir):
		os.mkdir(saving_dir)
	model_save_path = saving_dir + "/DDPGfD_kinovaGrip_{}".format(datetime.datetime.now().strftime("%m_%d_%y_%H%M"))

	# Initialize OU noise
	noise = OUNoise(4)
	noise.reset()
	expl_noise = OUNoise(4, sigma=0.001)
	expl_noise.reset()

	# Initialize SummaryWriter
	writer = SummaryWriter(logdir="./kinova_gripper_strategy/{}_{}/".format(args.policy_name, args.tensorboardindex))

	# Pretrain (No pretraining without imitation learning)
	# print("---- Pretraining ----")
	# num_updates = 1000
	# for k in range(int(num_updates)):
	# 	policy.train(replay_buffer, env._max_episode_steps)

	# Check and create directory
	trplot_saving_dir = "./train_plots"
	if not os.path.isdir(trplot_saving_dir):
		os.mkdir(trplot_saving_dir)

	# Heatmap initial object coordinates
	strain_obj_posx = np.array([])
	ftrain_obj_posx = np.array([])
	strain_obj_posy = np.array([])
	ftrain_obj_posy = np.array([])

	seval_obj_posx = np.array([])
	seval_obj_posy = np.array([])
	feval_obj_posx = np.array([])
	feval_obj_posy = np.array([])
	total_evalx = np.array([])
	total_evaly = np.array([])

	print("---- RL training in process ----")
	for t in range(int(args.max_episode)):
		env = gym.make(args.env_name)
		episode_num += 1

		# Fill training object list using latin square
		if (env.check_obj_file_empty("objects.csv")):
			env.Generate_Latin_Square(args.max_episode,"objects.csv",shape_keys)
		state, done = env.reset(env_name="env",shape_keys=requested_shapes,hand_orientation=requested_orientation,mode=args.mode), False

		noise.reset()
		expl_noise.reset()
		episode_reward = 0
		print("*** Episode Num: ",episode_num)

		obj_coords = env.get_obj_coords()

		while not done:
			# if t < args.start_timesteps:
			# 	action = env.action_space.sample()
			# else:
			action = (
				policy.select_action(np.array(state[0:48]))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

			# Select action randomly or according to policy
			# action = (policy.select_action(np.array(state)) + expl_noise.noise()).clip(-max_action, max_action)

			# Perform action obs, total_reward, done, info
			next_state, reward, done, info = env.step(action)
			done_bool = float(done) # if episode_timesteps < env._max_episode_steps else 0

			# Store data in replay buffer
			replay_buffer.add_wo_expert(state[0:48], action, next_state[0:48], reward, done_bool)
			if(info["lift_reward"] > 0):
				lift_success = True
			else:
				lift_success = False

			state = next_state
			episode_reward += reward

		# Train agent after collecting sufficient data:
		if episode_num > 10:
			for learning in range(100):
				actor_loss, critic_loss, critic_L1loss, critic_LNloss = policy.train(replay_buffer, env._max_episode_steps)

		# Heatmap postion data, get starting object position
		if(lift_success):
			x_val = obj_coords[0]
			y_val = obj_coords[1]
			x_val = np.asarray(x_val).reshape(1)
			y_val = np.asarray(y_val).reshape(1)

			strain_obj_posx = np.append(strain_obj_posx,x_val)
			strain_obj_posy = np.append(strain_obj_posy,y_val)

		else:
			x_val = obj_coords[0]
			y_val = obj_coords[1]
			x_val = np.asarray(x_val).reshape(1)
			y_val = np.asarray(y_val).reshape(1)

			ftrain_obj_posx = np.append(ftrain_obj_posx,x_val)
			ftrain_obj_posy = np.append(ftrain_obj_posy,y_val)

		# Evaluation and recording data for tensorboard
		if (t + 1) % args.eval_freq == 0:
			eval_ret = eval_policy(policy, args.env_name, args.seed, requested_shapes, requested_orientation, mode=args.mode, eval_episodes=200, compare=True)

			avg_reward = eval_ret[0]
			num_success = eval_ret[1]
			seval_obj_posx = np.append(seval_obj_posx,eval_ret[2])
			seval_obj_posy = np.append(seval_obj_posy,eval_ret[3])
			feval_obj_posx = np.append(feval_obj_posx,eval_ret[4])
			feval_obj_posy = np.append(feval_obj_posy,eval_ret[5])
			total_evalx = np.append(total_evalx,eval_ret[6])
			total_evaly = np.append(total_evaly,eval_ret[7])

			writer.add_scalar("Episode reward, Avg. 200 episodes",avg_reward, episode_num)
			writer.add_scalar("Actor loss", actor_loss, episode_num)
			writer.add_scalar("Critic loss", critic_loss, episode_num)
			writer.add_scalar("Critic L1loss", critic_L1loss, episode_num)
			writer.add_scalar("Critic LNloss", critic_LNloss, episode_num)
			evaluations.append(avg_reward)
			np.save("./results/%s" % (file_name), evaluations)
			print()

		# Save the x,y coordinates for object starting position (success vs failed grasp and lift)
		evplot_saving_dir = "./eval_plots"
		# Save coordinates every 1000 episodes
		if (t + 1) % 1000 == 0:
			save_coordinates(seval_obj_posx,seval_obj_posy,evplot_saving_dir+"/heatmap_eval_success"+str(episode_num))
			save_coordinates(feval_obj_posx,feval_obj_posy,evplot_saving_dir+"/heatmap_eval_fail"+str(episode_num))
			save_coordinates(total_evalx,total_evaly,evplot_saving_dir+"/heatmap_eval_total"+str(episode_num))
			seval_obj_posx = np.array([])
			seval_obj_posy = np.array([])
			feval_obj_posx = np.array([])
			feval_obj_posy = np.array([])
			total_evalx = np.array([])
			total_evaly = np.array([])

	train_totalx = np.append(strain_obj_posx,ftrain_obj_posx)
	train_totaly = np.append(strain_obj_posy,ftrain_obj_posy)

	# Save object postions from training
	save_coordinates(strain_obj_posx,strain_obj_posy,trplot_saving_dir+"/heatmap_train_success_new")
	save_coordinates(ftrain_obj_posx,ftrain_obj_posy,trplot_saving_dir+"/heatmap_train_fail_new")
	save_coordinates(train_totalx,train_totaly,trplot_saving_dir+"/heatmap_train_total_new")

	print("Saving into {}".format(model_save_path))
	policy.save(model_save_path)
