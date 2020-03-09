# This script is to teleoperate Kinova gripper in the mujoco env

import gym
import numpy as np
import pdb
import serial

env = gym.make('gym_kinova_gripper:kinovagripper-v0')
env.reset()

# setup serial
# ser = serial.Serial("/dev/ttyACM0", 9600)
# prev_action = [0.0,0.0,0.0,0.0]
action = np.array([0.3, 0.0, 0.0, 0.0])
t = 0
while True:
	obs, reward, done, _ = env.step(action)
	env.render()
