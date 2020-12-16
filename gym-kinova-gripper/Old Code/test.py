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
from expert_data import generate_Data, store_saved_data_into_replay
env.reset()


# setup serial
# ser = serial.Serial("/dev/ttyACM0", 9600)
# prev_action = [0.0,0.0,0.0,0.0]
action = np.array([0, 0.3, 0.3, 0.3])
t = 0
while True:
	#env = gym.make('gym_kinova_gripper:kinovagripper-v0')
	env.reset()

	for i in range(150):

	# read action from pyserial
	# curr_action = ser.readline().decode('utf8').strip().split(",")
	# for i in range(4):
	# 	curr_action[i] = float(curr_action[i])

	# if np.max(np.array(prev_action) - np.array(curr_action)) < 0.01:
	# 	# keep going
	# 	obs, reward, done, _ = env.step(prev_action)
	# else:
	# 	# update action
	# 	obs, reward, done, _ = env.step(curr_action)

	# print((curr_action))
	# prev_action = curr_action
		obs, reward, done, _ = env.step(action)
	# action[1] += 0.5
	# action[2] += 0.2
	# action[3] += 0.7
		env.render()
	# if t > 25:
	# 	action = np.array([0.1, 0.8, 0.8, 0.8])
	# # print()
	# t += 1
    finger=[0.3,0.1,0.1,0.1]
    for i in range(50):
        obs, reward, done, _ = env.step(action)
	# action[1] += 0.5
	# action[2] += 0.2
	# action[3] += 0.7
		env.render()
        