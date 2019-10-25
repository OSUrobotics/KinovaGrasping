# from envs.kinova_gripper_env import KinovaGripper_Env
# env.action_space.sample()
import gym
from gym import spaces
import numpy as np
import pdb
import pickle
import utils

env = gym.make('gym_kinova_gripper:kinovagripper-v0')
from ounoise import OUNoise
from expert_data import generate_Data
env.reset()

reward_total = 0

step = 0
# file_name = open("data_cube_5_10_07_19_1612.pkl", "rb")
# data = pickle.load(file_name)
# states = np.array(data["states"])
# random_states_index = np.random.randint(0, len(states), size = len(states))
# global local = 48 , metric 47
num_episode = 5000
filename = "data_cube_9"
replay_buffer = utils.ReplayBuffer_episode(48, 4, 100, 100)
replay_buffer = generate_Data(env, 10, "random", replay_buffer)
'''
for _ in range(3):
	# noise.reset()
	env.reset()
	# state = env.env.intermediate_state_reset(states[np.random.choice(random_states_index, 1)[0]])

	done = False
	finger = np.array([0.0, 0.5, 0.5, 0.5])
	i = 0
	while not done:

		i += 1
		obs, reward, done, _ = env.step(finger)

		reward_total += reward 

		# print(reward)
		if i > 20:
			finger = np.array([0.0, 0.5,0.5, 0.5])
		if i > 60:
			finger = np.array([2.0, 0.5,0.5, 0.5])

'''