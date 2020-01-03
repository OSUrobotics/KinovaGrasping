#!/usr/bin/env python3

"""
Author : Yi Herng Ong
Title : Script to collect data from robot at random joint configuration 
"""

# Set object at random close palm position
# Set random joint configuration for each object offsets
# Close fingers for 5 time steps, then lift
# Record whether it's success or fail

import gym
import numpy as np
import pdb

def PID(target, current):
	err = target - current
	if err < 0.0:
		err = 0.0
	diff = err / 4
	vel = err + diff # PD control
	action = (vel / 0.8) * 0.3 
	
	return action

def getRandomJoint(obj_size, obj_pose):
	if obj_size == "s":
		if obj_pose < 0.0:
			f2 = np.random.uniform(0.1, 0.5)
			f3 = np.random.uniform(0.1, 0.5)
			f1 = np.random.uniform(0.1, 0.5)
		else:
			f2 = np.random.uniform(0.1, 0.5)
			f3 = np.random.uniform(0.1, 0.5)
			f1 = np.random.uniform(0.1, 0.21) 

	return np.array([f1, 0.5, 0.5])

env = gym.make('gym_kinova_gripper:kinovagripper-v0')
for i in range(1):
	obs = env.reset()

	target_joint_config = getRandomJoint("s", obs[21])
	# print(target_joint_config)
	step = 0
	# for _ in range(200):
	reach = False
	while True:
		# finger action 
		finger_action = []
		# first part go to random joint config 		
		# print((np.array(obs[25:28])))
		if (np.max(np.abs((np.array(obs[25:28]) - target_joint_config))) > 0.01) and (reach == False):
			for finger in range(3):
				# print(target_joint_config[i], obs[25+i])
				finger_action.append(PID(target_joint_config[finger], obs[25+finger]))
			# print(finger_action)

			action = np.array([0.0, finger_action[0], finger_action[1], finger_action[2]])
		# Second part
		else:
			# pdb.set_trace()
			reach = True
			step += 1
			if step > 50 and step < 100: # wait for one second
				action = np.array([0.0, 0.3, 0.15, 0.15])
			elif step > 100:
				action = np.array([0.3, 0.3, 0.15, 0.15])
			else:
				action = np.array([0.0, 0.0, 0.0, 0.0])

		print(step)
		obs, reward, done, _ = env.step(action)
		env.render()


		



# env.reset()
# action = np.array([0.0, 0.1, 0.1, 0.1])
# while True:
# 	obs, reward, done, _ = env.step(action)
# 	print(np.abs(np.max(np.array(obs[25:28]) - 0.8)))
# 	if np.abs(np.max(np.array(obs[25:28]) - 0.8)) < 0.01:
# 		action = np.array([0.0, 0.0, 0.0, 0.0])
# 	env.render()
