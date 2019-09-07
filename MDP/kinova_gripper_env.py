#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Kinova 3-fingered gripper in mujoco environment
# Summer 2019

###############


import gym
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import numpy as np
from mujoco_py import MjViewer, load_model_from_path, MjSim
from PID_Kinova_MJ import *
import math
import matplotlib.pyplot as plt
import time
import os, sys

# resolve cv2 issue 
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

class KinovaGripper_Env(gym.Env):
	def __init__(self, arm_or_end_effector, frame_skip, state_rep):
		file_dir = os.path.dirname(os.path.realpath(__file__))
		if arm_or_end_effector == "arm":
			self._model = load_model_from_path(file_dir + "/kinova_description/j2s7s300.xml")
			full_path = file_dir + "/kinova_description/j2s7s300.xml"
		elif arm_or_end_effector == "hand":
			self._model = load_model_from_path(file_dir + "/kinova_description/j2s7s300_end_effector.xml")
			full_path = file_dir + "/kinova_description/j2s7s300_end_effector.xml"
		else:
			print("CHOOSE EITHER HAND OR ARM")
			raise ValueError
		# self._model = load_model_from_path("/home/graspinglab/NearContactStudy/MDP/jaco/jaco.xml")
		
		self._sim = MjSim(self._model)
		
		self._viewer = MjViewer(self._sim)

		self._timestep = self._sim.model.opt.timestep
		# print(self._timestep)
		# self._sim.model.opt.timestep = self._timestep

		self._torque = [0,0,0,0]
		self._velocity = [0,0,0,0]

		self._jointAngle = [0,0,0,0]
		self._positions = [] # ??
		self._numSteps = 0
		self._simulator = "Mujoco"

		# define pid controllers for all joints
		self.pid = [PID_(65,0.04,0.0), PID_(10,0.01,0.0), PID_(10,0.01,0.0), PID_(10,0.01,0.0)]
		# self.pid = [PID_(10,0.01,0.0), PID_(1,0.0,0.0), PID_(1,0.0,0.0), PID_(1,0.0,0.0)]

		# Parameters for cost function
		self.state_des = 0.20 
		self.initial_state = np.array([0.0, 0.0, 0.0, 0.0])
		# mujoco_env.MujocoEnv.__init__(self, full_path, frame_skip)
		self.frame_skip = frame_skip
		self.state_rep = state_rep


	def set_step(self, seconds):
		self._numSteps = seconds / self._timestep
		# print(self._numSteps)

	# might want to do this function on other file to provide command on ros moveit as well
	def set_target_thetas(self, thetas): 
		self.pid[0].set_target_jointAngle(thetas[0])
		self.pid[1].set_target_jointAngle(thetas[1])
		self.pid[2].set_target_jointAngle(thetas[2])
		self.pid[3].set_target_jointAngle(thetas[3])


	def _finger_control(self):
		for i in range(3):
			self._jointAngle[i+1] = self._sim.data.sensordata[i+1]
			self._torque[i+1] = self.pid[i+1].get_Torque(self._jointAngle[i+1])
			self._sim.data.ctrl[i+1] = self._torque[i+1]

	def _wrist_control(self):
		self._jointAngle[0] = self._sim.data.sensordata[0]
		self._torque[0] = self.pid[0].get_Torque(self._jointAngle[0])
		self._sim.data.ctrl[0] = self._torque[0]


	# get 3D transformation matrix of each joint
	def _get_trans_mat(self, joint_geom_name):
		finger_joints = joint_geom_name	
		finger_pose = []
		empty = np.array([0,0,0,1])
		for each_joint in finger_joints:
			arr = []
			for axis in range(3):
				temp = np.append(self._sim.data.get_geom_xmat(each_joint)[axis], self._sim.data.get_geom_xpos(each_joint)[axis])
				arr.append(temp)
			arr.append(empty)
			arr = np.array(arr)
			finger_pose.append(arr)	
			# print(each_joint, finger_pose)


		return np.array(finger_pose)

	# return global or local transformation matrix
	def _get_finger_pose(self, local_or_global):
		palm = self._get_trans_mat(["palm"])[0]
		# print(palm)
		finger_joints = self._get_trans_mat(["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"])

		if local_or_global == "global":
			return finger_joints

		elif local_or_global == "local":
			finger_joints_local = []
			palm_inverse = np.linalg.inv(palm)
			for joint in range(len(finger_joints)):
				joint_in_local_frame = np.matmul(finger_joints[joint], palm_inverse)
				finger_joints_local.append(joint_in_local_frame)

			return finger_joints_local

		else:
			print("Wrong entry, neither global or local frame")
			raise ValueError

	# get range data from 1 step of time 
	# Uncertainty: rangefinder could only detect distance to the nearest geom, therefore it could detect geom that is not object
	def _get_rangefinder_data(self):
		range_data = []
		for i in range(14):
			range_data.append(self._sim.data.sensordata[i+7])

		return np.array(range_data)

	def _get_obj_pose(self):
		arr = self._sim.data.get_geom_xpos("cube")
		arr = np.append(arr, 0)
		return arr


	def _get_done(self):
		return False

	# set reward as object being lifted 
	# need normalize
	def _get_dist_reward(self):
		state = self._get_obj_pose()
		reward = -(self.state_des - state[2]) # only get z for now

		return reward

	def _get_norm_reward(self):
		state = self._get_obj_pose()
		reward_norm = (state[2] - 0.06) / (0.2 - 0.06)
		if reward_norm > 1.0:
			reward_norm = 1.0
		elif reward_norm < 0.0:
			reward_norm = 0.0

		return reward_norm


	def _get_reward_based_on_palm(self, options):
		pc = self._get_palm_center(options)
		obj_state = self._get_obj_pose()
		if options == "world" or options == "metric":
			reward = - abs(pc[0] - obj_state[0]) - abs(pc[1] - obj_state[1])

		elif options == "palm":
			obj_trans_mat = self._get_trans_mat(["cube"])[0]
			palm_trans_mat = self._get_trans_mat(["palm"])[0]
			palm_inverse = np.linalg.inv(palm_trans_mat)
			obj_local_mat = np.matmul(palm_inverse, obj_trans_mat)
			obj_x_local = obj_local_mat[0][3]
			obj_y_local = obj_local_mat[1][3]
			reward = - abs(pc[0] - obj_x_local) - abs(pc[1] - obj_y_local)

		return reward


	def _get_palm_center(self, options):
		if options == "world" or options == "metric":
			return [0.0, 0.0]
		elif options == "palm":
			return [0.0, 0.119]
		else:
			raise ValueError


	def _get_joint_states(self):
		arr = []
		for i in range(6):
			arr.append(self._sim.data.sensordata[i+1])

		return np.array(arr) 

	def _get_state(self):
		return np.array([self._sim.data.qpos[0], self._sim.data.qpos[1], self._sim.data.qpos[3], self._sim.data.qpos[5]]) 

	# only set proximal joints, cuz this is an underactuated hand
	def _set_state(self, states):
		self._sim.data.qpos[0] = states[0]
		self._sim.data.qpos[1] = states[1]
		self._sim.data.qpos[3] = states[2]
		self._sim.data.qpos[5] = states[3]
		self._sim.data.set_joint_qpos("cube", [states[4], states[5], states[6], 1.0, 0.0, 0.0, 0.0])
		self._sim.forward()

	def _reset_state(self):
		self._set_state(self.initial_state)


	def _get_obs(self):
		jA = self._get_joint_states()
		obj_pose = self._get_obj_pose()
		if self.state_rep == "world":
			print("here")
			fpose = list(self._get_finger_pose("global"))
			fpose.append(obj_pose)
			print (fpose)
			# print("obj_pose", obj_pose )
			return fpose
		elif self.state_rep == "palm":
			return np.concatenate((self._get_finger_pose("local"), jA, obj_pose))
		elif self.state_rep == "rangedata":
			return np.concatenate((self._get_rangefinder_data(), jA, obj_pose))


	# each step will last for 0.5 second
	def step(self, action, render=False):
		# print("control range", self._model.actuator_ctrlrange.copy())
		curr_handpose = np.array([0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.08])

		self._set_state(curr_handpose)

		self.set_target_thetas(curr_handpose)
		step = 0
		start = time.time()

		total_reward = 0

		initial_state = self._get_obs()

		for i in range(self.frame_skip):

			target = np.array(action) # rad 
			target_vel = np.zeros(3) + target
			target_delta = target / 500
			# print(np.max(np.abs(gripper[1:] - target_vel)) > 0.01)
			# if np.max(np.abs(gripper[1:] - target_vel)) > 0.001:
				# print("curling in")
			if self._sim.data.time < 0.5:
				curr_handpose[1:4] += target_delta
				self.set_target_thetas(curr_handpose[0:4])

			else: # grasp validation
				wrist_target = 0.2
				wrist_delta = wrist_target / 500
				if abs(curr_handpose[0] - wrist_target) > 0.001:
					curr_handpose[0] += wrist_delta
					self.set_target_thetas(curr_handpose[0:4])

			self._wrist_control() # wrist action
			self._finger_control() # finger action
			# step += 1
			self._sim.step() # update every 0.002 seconds (500 Hz)
			if render:
				self._viewer.render()
			# print(self._sim.data.qpos[:])
			total_reward += self._get_reward_based_on_palm("world")
				

		curr_state = self._get_obs()
		# print(curr_state)
		obs = np.concatenate((curr_state, initial_state))

			# pose = 
		if total_reward < -2:
			done = True
		else:
			done = self._get_done()
		
		return obs, total_reward, done, {}

if __name__ == '__main__':
	# print(os.path.dirname(os.path.realpath(__file__)))
	sim = KinovaGripper_Env("hand", frame_skip=250, state_rep="world")
	# data = sim.physim_mj()
	sim.step([0.7, 0.3, 0.5], True)
