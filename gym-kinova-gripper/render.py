#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Learn finger control strategy 
# Summer 2019

###############
import os, sys
import numpy as np
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2, A2C
import matplotlib.pyplot as plt

n_cpu = 2
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # to import cv2 successfully at stable_baselines
# env = SubprocVecEnv([lambda: gym.make('gym_kinova_gripper:kinovagripper-v0' )])

env = DummyVecEnv([lambda: gym.make('gym_kinova_gripper:kinovagripper-v0')])


model = PPO2.load("ppo2_kinova_strategy_learn_movebox_3")
# model = PPO2.load("ppo2_kinova_strategy_ec01_lr0001_steps2e5")

obs = env.reset()
qvel = []
print(env.action_space.high)
for _ in range(50):
	# action, _states = model.predict(obs, deterministic=True)
	actions,_, states, neglogp = model.step(obs, deterministic=True)
	# value = model.proba_step(obs)
	# value = model.value(obs)
	# print("states", value)
	obs, rewards, dones, info = env.step(actions)
	# print(env._sim.data.get_joint_qvel("j2s7s300_joint_finger_1"))
	# print("rewards", rewards)
	# env.render()

	# print(model.action_probability(obs))
	# print(actions)

# 	qvel.append(actiona[0][0])

# timestep = np.arange(0, len(qvel))
# print(min(map(abs, qvel)))
# plt.plot(timestep, qvel)
# plt.show()