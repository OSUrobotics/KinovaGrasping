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
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

n_cpu = 4
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # to import cv2 successfully at stable_baselines
env = SubprocVecEnv([lambda: gym.make('gym_kinova_gripper:kinovagripper-v0') for i in range(n_cpu)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps = 25000)
model.save("ppo2_kinova_strategy")


