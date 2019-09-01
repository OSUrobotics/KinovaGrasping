#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Kinova 3-fingered gripper in mujoco environment
# Summer 2019

###############


import gym
import numpy as np
from mujoco_py import MjViewer, load_model_from_path, MjSim
from PID_Kinova_MJ import *
import math
import matplotlib.pyplot as plt
import time
import os

class Kinova_MJ(object):
	def __init__(self, arm_or_end_effector):
		file_dir = os.path.dirname(os.path.realpath(__file__))
		if arm_or_end_effector == "arm":
			self._model = load_model_from_path(file_dir + "/kinova_description/j2s7s300.xml")
		elif arm_or_end_effector == "hand":
			self._model = load_model_from_path(file_dir + "/kinova_description/j2s7s300_end_effector.xml")
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
	def _get_joint_pose(self, joint_geom_name):
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
		palm = self._get_joint_pose(["palm"])[0]
		# print(palm)
		finger_joints = self._get_joint_pose(["f1_prox", "f1_dist", "f2_prox", "f2_dist", "f3_prox", "f3_dist"])

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
			range_data.append(self._sim.data.sensordata[i+4])

		return np.array(range_data)

	def _get_obj_pose(self):
		return self._sim.data.get_geom_xpos("cube")


	# def _get_reward(self, state, action):


	def step(self, action, render=False):
		initial_handpose = np.zeros(7)
		self._sim.data.qpos[0:7] = initial_handpose 
		initial_fingerpose = np.array([0.0, 0.0, 0.0, 0.0])
		# print(len(initial_fingerpose))
		gripper = np.array([0.0, 0.0, 0.0, 0.0])
		self.set_target_thetas(initial_fingerpose)
		step = 0
		start = time.time()


		for _ in range(500):
			# print(self._sim.data.sensordata[:])
			# print("f1p:",self._sim.data.get_geom_xpos("f2_prox"))
			# print("object:",type(self._sim.data.get_geom_xpos("cube")))
			# print(np.append(self._sim.data.get_geom_xmat("f1_prox")[0], self._sim.data.get_geom_xpos("f1_prox")[0]))
			pose = self._get_finger_pose("global")
			
			# print("here:",self._sim.data.get_site_xpos("f1_prox")[:])
			# print("here:",self._sim.data.sensordata[4],self._sim.data.sensordata[5], self._sim.data.sensordata[6], self._sim.data.sensordata[7], self._sim.data.sensordata[8] )
			range_data = self._get_rangefinder_data()

			self._wrist_control()
			self._finger_control()
			# if step > 1000:				
			target = np.array(action) # rad 
			target_vel = np.zeros(3) + target
			target_delta = target / 500
			# print(np.max(np.abs(gripper[1:] - target_vel)) > 0.01)
			if np.max(np.abs(gripper[1:] - target_vel)) > 0.001:
				# print("curling in")
				gripper[1:] += target_delta
				self.set_target_thetas(gripper)
			else: # grasp validation
				wrist_target = 0.2
				wrist_delta = wrist_target / 500
				if abs(gripper[0] - wrist_target) > 0.001:
					gripper[0] += wrist_delta
					self.set_target_thetas(gripper)
				else:
					print("time_spent:", time.time() - start)		

			step += 1
			self._sim.step()
			if render:
				self._viewer.render()





if __name__ == '__main__':
	print(os.path.dirname(os.path.realpath(__file__)))
	sim = Kinova_MJ("hand")
	# data = sim.physim_mj()
	sim.step([0.8, 0.8, 0.8], True)
