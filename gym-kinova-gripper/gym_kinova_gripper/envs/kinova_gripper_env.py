#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Kinova 3-fingered gripper in mujoco environment
# Summer 2019

###############


import gym
from gym import utils, spaces
from gym.utils import seeding
# from gym.envs.mujoco import mujoco_env
import numpy as np
from mujoco_py import MjViewer, load_model_from_path, MjSim
# from PID_Kinova_MJ import *
import math
import matplotlib.pyplot as plt
import time
import os, sys
from scipy.spatial.transform import Rotation as R
import random


# resolve cv2 issue 
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# frame skip = 20
# action update time = 0.002 * 20 = 0.04
# total run time = 40 (n_steps) * 0.04 (action update time) = 1.6

class KinovaGripper_Env(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self, arm_or_end_effector="hand", frame_skip=4, state_rep="global"):
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
		self.all_states = None
		# self.action_space = spaces.Box(low=np.array([-0.8, -0.8, -0.8]), high=np.array([0.8, 0.8, 0.8]), dtype=np.float32)
		# self.action_space = ac_space[0]
		self.action_space = spaces.Box(low=np.array([-0.50]), high=np.array([0.50]), dtype=np.float64)

		# print()
		if self.state_rep == "global" or self.state_rep == "local":
			# obs_min = np.array([-0.1, -0.1, 0.0, -360, -360, -360, -0.1, -0.1, 0.0, -360, -360, -360,
			# 	-0.1, -0.1, 0.0, -360, -360, -360,-0.1, -0.1, 0.0, -360, -360, -360,
			# 	-0.1, -0.1, 0.0, -360, -360, -360,-0.1, -0.1, 0.0, -360, -360, -360, 
			# 	-0.1, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
			obs_min = np.array([-0.1, -0.1, 0.0, -360, -360, -360, -0.1, -0.1, 0.0, -360, -360, -360])
		
			# obs_max = np.array([0.1, 0.1, 0.3, 360, 360, 360, 0.1, 0.1, 0.3, 360, 360, 360,
			# 	0.1, 0.1, 0.3, 360, 360, 360,0.1, 0.1, 0.3, 360, 360, 360,
			# 	0.1, 0.1, 0.3, 360, 360, 360,0.1, 0.1, 0.3, 360, 360, 360,
			# 	0.1, 0.7, 0.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
			obs_max = np.array([0.1, 0.1, 0.3, 360, 360, 360, 0.1, 0.1, 0.3, 360, 360, 360])

			self.observation_space = spaces.Box(low=obs_min , high=obs_max, dtype=np.float32)
		elif self.state_rep == "metric":
			obs_min = np.zeros(17) 
			obs_max = obs_min + np.Inf
			self.observation_space = spaces.Box(low=obs_min , high=obs_max, dtype=np.float32)


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


	# def _get_action_space(self, action_min=0.0, action_max=2.0):


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


	def _get_local_pose(self, mat):
		rot_mat = []
		trans = []
		# print(mat)
		for i in range(3):
			orient_temp = []

			for j in range(4):
				if j != 3:
					orient_temp.append(mat[i][j])
				elif j == 3:
					trans.append(mat[i][j])
			rot_mat.append(orient_temp)

		# print()
		r = R.from_dcm(rot_mat)
		euler_vec = r.as_euler('zyx', degrees=True)
		pose = list(trans) + list(euler_vec)

		return pose

	# return global or local transformation matrix
	def _get_obs(self):
		palm = self._get_trans_mat(["palm"])[0]
		# print(palm)
		# finger_joints = ["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
		finger_joints = ["f1_prox", "f1_dist"]
		finger_joints_transmat = self._get_trans_mat(["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"])
		fingers_6D_pose = []
		if self.state_rep == "global":
			# return finger_joints
			# print (self._sim.data.get_geom_xmat("f1_prox"))				
			for joint in finger_joints:
				rot_mat = R.from_dcm(self._sim.data.get_geom_xmat(joint))
				euler_vec = rot_mat.as_euler('zyx', degrees=True)
				trans = self._sim.data.get_geom_xpos(joint)
				trans = list(trans)
				trans += list(euler_vec)
				for i in range(6):
					fingers_6D_pose.append(trans[i])

			# return np.array(fingers_6D_pose)

		elif self.state_rep == "local":
			finger_joints_local = []
			palm_inverse = np.linalg.inv(palm)
			for joint in range(len(finger_joints_transmat)):
				joint_in_local_frame = np.matmul(finger_joints_transmat[joint], palm_inverse)
				pose = self._get_local_pose(joint_in_local_frame)
				for i in range(6):
					fingers_6D_pose.append(pose[i])
				# finger_joints_local.append(joint_in_local_frame)

			# return fingers_6D_pose)
		elif self.state_rep == "metric":
			fingers_6D_pose = self._get_rangefinder_data()

		else:
			print("Wrong entry, enter one of the following: global, local, metric")
			raise ValueError

		# print(len(fingers_6D_pose))
		# if self.state_rep != "metric":
		# 	obj_pose = self._get_obj_pose()

		# 	joint_states = self._get_joint_states()
		# 	fingers_6D_pose = fingers_6D_pose + list(obj_pose) + joint_states

		return fingers_6D_pose 


	# get range data from 1 step of time 
	# Uncertainty: rangefinder could only detect distance to the nearest geom, therefore it could detect geom that is not object
	def _get_rangefinder_data(self):
		range_data = []
		for i in range(17):
			range_data.append(self._sim.data.sensordata[i+7])

		return range_data

	def _get_obj_pose(self):
		arr = self._sim.data.get_geom_xpos("cube")
		# arr = np.append(arr)
		return arr


	def _get_done(self):
		return False


	# this reward min is 0.932 max is 1
	def _get_dot_product(self):
		obj_state = self._get_obj_pose()
		hand_pose = self._sim.data.get_body_xpos("j2s7s300_link_7")
		obj_state_x = abs(obj_state[0] - hand_pose[0])
		obj_state_y = abs(obj_state[1] - hand_pose[1])
		obj_vec = np.array([obj_state_x, obj_state_y])
		obj_vec_norm = np.linalg.norm(obj_vec)
		obj_unit_vec = obj_vec / obj_vec_norm

		center_x = abs(0.0 - hand_pose[0])
		center_y = abs(0.0 - hand_pose[1])
		center_vec = np.array([center_x, center_y])
		center_vec_norm = np.linalg.norm(center_vec)
		center_unit_vec = center_vec / center_vec_norm
		# print(obj_unit_vec, center_unit_vec)
		# print(obj_vec, center_vec)
		dot_prod = np.dot(obj_unit_vec, center_unit_vec)
		# print(math.degrees(np.arccos(dot_prod)))

		return dot_prod**10 # cuspy to get distinct reward

	def _get_contact_distance(self):
		finger_pose = self._sim.data.get_geom_xpos("f2_dist")
		x = finger_pose[0]
		y = finger_pose[1]
		obj_state = self._get_obj_pose()
		return x, obj_state[0]

	'''
	Reward function
	'''
	def _get_reward(self, state, action):
		# obj_state = self._get_obj_pose()

		# reward = - abs(obj_state[0] - 0.0) - abs(obj_state[1] - 0.0)

		# reward = self._get_dot_product() 
		
		f1 = self._sim.data.get_geom_xpos("f1_dist")
		# f2 = self._sim.data.get_geom_xpos("f2_dist")
		# f3 = self._sim.data.get_geom_xpos("f3_dist")

		f1_curr = (0.05813983 - state[0])**2 + (0.01458329 - state[1])**2
		# f1_curr = math.sqrt((0.05813983 - state[0])**2 + (0.01458329 - state[1])**2)
		# f1_curr = (0.058 - state[0])**2 + (0.014 - state[1])**2


		# f1_dist = math.sqrt((0.07998454 - 0.05813983)**2 + (0.03696302 - 0.01458329)**2)
		# x = f1_curr / f1_dist # normalize between 0 and 1
		# # print(x)
		# if x > 1:
		# 	x = 1
		# reward = math.sqrt(1 - x**2) - 1

		# f1_init = math.sqrt((0.07998454 - f1[0])**2 + (0.03696302 - f1[1])**2)

		# f1_reward = (math.exp(-100*f1_curr))

		f1_reward = 11*(math.exp(-100*f1_curr) - 1)

		# f2_curr = math.sqrt((-0.06437128 - f2[0])**2 + abs(0.02180294 - f2[1])**2)
		# f2_dist = math.sqrt((-0.07601888 - -0.06437128)**2 + (0.03679844 - 0.02180294)**2)

		# f2_reward = (math.exp(-100*f2_curr))

		# f3_curr = math.sqrt((-0.06448944 - f3[0])**2 + abs(0.02191481 - f3[1])**2)
		# f3_dist = math.sqrt((-0.07606889 - -0.06448944)**2 + (0.03684846 - 0.02191481)**2)

		# f3_reward = (math.exp(-100*f3_curr))
		# print("f3 reward", f3_curr)

		# reward = (f1_reward + f2_reward + f3_reward) / 3.0

		if f1_curr < 1e-6: 
			reward = f1_reward + (-100*action**2)
		else:
			reward = f1_reward

		# target = 0.5
		# err = (target - state)**2
		# reward = 5*(math.exp(-err) -1)

		return reward


	def _get_joint_states(self):
		arr = []
		for i in range(6):
			arr.append(self._sim.data.sensordata[i+1])

		return arr # it is a list

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

	def _get_obj_size(self):
		return self._sim.model.geom_size[-1]


	def _set_obj_size(self, random_type=False, chosen_type="box", random_size=False):
		self.hand_param = {}
		self.hand_param["span"] = 0.175
		self.hand_param["depth"] = 0.08
		self.hand_param["height"] = 0.20 # including distance between table and hand

		geom_type = {"box": [6, 3], "cylinder": [5, 3], "sphere": [2, 1]}
		geom_index = 0
		if random_type:
			chosen_type = random.choice(list(geom_type.keys()))
			geom_index = geom_type[chosen_type][0]
			geom_dim_size = geom_type[chosen_type][1]

		else:
			geom_index = geom_type[chosen_type][0]
			geom_dim_size = geom_type[chosen_type][1]

		width_max = self.hand_param["span"] * 0.35
		width_min = self.hand_param["span"] * 0.10

		height_max = self.hand_param["height"] 
		height_min = self.hand_param["height"] * 0.25

		geom_dim = np.array([])

		if random_size:
			if geom_index == 6 or geom_index == 5:
				# geom_dim = []
				width = random.uniform(width_min, width_max)
				height = random.uniform(height_min, height_max)
				geom_dim = np.array([width, width, height])
			elif geom_index == 2:
				radius = random.uniform(width_min, width_max)
				while radius < height_min:
					radius = random.uniform(width_min, width_max)
				geom_dim = np.array([radius])
			else:
				raise ValueError

			return geom_index, geom_dim
		else:
			# return medium size box
			width = (self.hand_param["span"] * 0.20) / 2
			height = 0.05

			geom_dim = np.array([width, width, height])

			return geom_index, geom_dim



	def reset(self):
		# geom_index, geom_dim = self._set_obj_size()
		# self._sim.model.geom_type[-1] = geom_index
		# self._sim.model.geom_size[-1] = geom_dim
		self.all_states = np.array([0.0, 0.0, 0.0, 0.0, -0.05, -0.2, 0.05])
		self._set_state(self.all_states)		
		states = self._get_obs()
		# obs_x = self._sim.data.get_geom_xpos("f1_dist")[0]
		# obs_y = self._sim.data.get_geom_xpos("f1_dist")[1]
		# states = np.array([obs_x, obs_y])				

		return states

	def forward(self):
		curr_allpose = np.array([0.0, 0.0, 0.0, 0.0, -0.07, -0.2, 0.05])
		self._set_state(curr_allpose)

		while True:
			self._sim.forward()
			self._viewer.render()

	def render(self, mode='human'):
		self._viewer.render()

	def close(self):
		pass


	def step(self, action, render=True):
		# print("control range", self._model.actuator_ctrlrange.copy())
		# curr_handpose = np.array([0.0, 0.0, 0.0, 0.0, -0.07, 0.003, 0.05])

		# step = 0
		# start = time.time()

		total_reward = 0

		# initial_state = self._get_obs()
		# self._sim.model.geom_type[-1] = 2
		# self._sim.model.geom_size[-1] = np.array([0.03])
		# qvel = []
		state = self._sim.data.get_geom_xpos("f1_dist")
		# state = self._sim.data.get_joint_qpos("j2s7s300_joint_finger_1")
		for _ in range(self.frame_skip):
			self._wrist_control() # wrist action
			# self._finger_control() # finger action

			for i in range(1):
				# if abs(action[i]) < 0.2:
				# 	self._sim.data.ctrl[i+1] = 0.0
				# else: 
				self._sim.data.ctrl[i+1] = 1.1*action[i] 


			# self._sim.data.ctrl[1], self._sim.data.ctrl[4] = self._pd_controller(action)

			# # joint position control
			# ref_poses, ref_vels = self._joint_position_controller(action)
			# for i in range(1):
			# 	self._sim.data.ctrl[i+1] = ref_poses[i]
			# 	self._sim.data.ctrl[i+4] = ref_vels[i]

			# joint velocity control
			# ref_forces, ref_vels = self._joint_velocity_controller(action)
			# for i in range(1):
			# 	self._sim.data.ctrl[i+1] = ref_forces[i]
			# 	self._sim.data.ctrl[i+4] = ref_vels[i]			

			self._sim.step() # update every 0.002 seconds (500 Hz)
			if render:
				self.render()
			# print(self._sim.data.get_joint_qvel("j2s7s300_joint_finger_1"))
			# print(self._sim.data.get_joint_qpos("j2s7s300_joint_finger_2"))
			# print(self._sim.data.get_joint_qpos("j2s7s300_joint_finger_3"))
			# if self._sim.data.time == 1:
			# qvel.append(
			# print(self._sim.data.get_joint_qvel("j2s7s300_joint_finger_1"))


		total_reward = self._get_reward(state,action[0])
		done = self._get_done()
		obs = self._get_obs()

		# obs_x = self._sim.data.get_geom_xpos("f1_dist")[0]
		# obs_y = self._sim.data.get_geom_xpos("f1_dist")[1]
		# obs = np.array([obs_x, obs_y])
		
		return obs, total_reward, done, {}


	def _joint_position_controller(self, action):
		# err = action - self._sim.data.get_joint_qpos("j2s7s300_joint_finger_1") 
		ref_pos = action
		ref_vel = action * self._sim.model.opt.timestep
		return ref_pos, ref_vel

	def _joint_velocity_controller(self, action):
		err = action - self._sim.data.get_joint_qpos("j2s7s300_joint_finger_1") 
		diff_err = err * self._sim.model.opt.timestep
		ref_force = err + diff_err
		ref_vel = action
		return ref_force, ref_vel


class PID_(object):
	def __init__(self, kp=0.0, kd=0.0, ki=0.0):
		self._kp = kp
		self._kd = kd
		self._ki = ki

		self._samplingTime = 0.0001 # param
		self._prevError = 0.0 
		self._targetjA = 0.0

		self.sum_error = 0.0
		self.diff_error = 0.0

	def set_target_jointAngle(self, theta):
		self._targetjA = theta

	def get_Torque(self, theta):
		error = self._targetjA - theta # might add absolute to compensate motion on the other side
		# print("error", error)
		# if error < 0.001:
		# 	print(True)
		self.sum_error += error*self._samplingTime
		self.diff_error = (error - self._prevError)/ self._samplingTime
		output_Torque = self._kp*error + self._ki*self.sum_error + self._kd*self.diff_error
		self._prevError = error
		# if output_Torque > 1:
		# 	output_Torque = 1
		# elif output_Torque < -1:
		# 	output_Torque = -1
		# print (output_Torque)
		return output_Torque 

	def get_Velocity(self, theta):
		error = self._targetjA - theta # might add absolute to compensate motion on the other side
		# print("error", error)
		self.sum_error += error*self._samplingTime
		self.diff_error = (error - self._prevError)/ self._samplingTime
		output_Vel = self._kp*error + self._ki*self.sum_error + self._kd*self.diff_error
		self._prevError = error
		if output_Vel > 30:
			output_Vel = 30
		elif output_Vel < -30:
			output_Vel = -30
		return output_Vel


# if __name__ == '__main__':
	# print(os.path.dirname(os.path.realpath(__file__)))
	# sim = KinovaGripper_Env("hand", frame_skip=250, state_rep="global")
	# data = sim.physim_mj()
	# sim.step([0.7, 0.5, 0.5], True)
	# sim.forward()

	# else: # grasp validation
	# 	wrist_target = 0.2
	# 	wrist_delta = wrist_target / 500
	# 	if abs(curr_handpose[0] - wrist_target) > 0.001:
	# 		curr_handpose[0] += wrist_delta
	# 		self.set_target_thetas(curr_handpose[0:4])
	# self._viewer.add_marker(pos=np.array([0.0, 0.07, 0.0654]), size=np.array([0.02, 0.02, 0.02]))

	# target = np.array(action) # rad 
	# target_vel = np.zeros(3) + action
	# target_vel = np.zeros(1) + action
	# target_delta = target_vel / 500

	# self.all_states[1:4] += target_delta
	# self.all_states[1] += target_delta
	# self.set_target_thetas(self.all_states[0:4])	