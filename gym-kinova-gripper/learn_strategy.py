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

n_cpu = 4
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # to import cv2 successfully at stable_baselines
# env = SubprocVecEnv([lambda: gym.make('gym_kinova_gripper:kinovagripper-v0') for i in range(n_cpu)])
env = DummyVecEnv([lambda: gym.make('gym_kinova_gripper:kinovagripper-v0') for i in range(n_cpu)])
# env = DummyVecEnv([lambda: gym.make('gym_kinova_gripper:kinovagripper-v0')])


# ent_coef=0.1, lr = 0.0001 14,15
# PPO2
model = PPO2(MlpPolicy, env, n_steps=50, learning_rate = 0.0002, verbose=1, tensorboard_log="./kinova_gripper_strategy" ,full_tensorboard_log=True)
model.learn(total_timesteps = 200000)
model.save("ppo2_kinova_strategy_learnf1_test_17") # 14, 15 are good for demo

# A2C
# model = A2C(MlpPolicy, env, learning_rate = 0.0002, verbose=1, tensorboard_log="./kinova_gripper_strategy" ,full_tensorboard_log=True)
# model.learn(total_timesteps = 200000)
# model.save("a2c_test1")