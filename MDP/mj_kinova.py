#!/usr/bin/env python3

###############
#Author: Yi Herng Ong
#Purpose: import kinova jaco j2s7s300 into Mujoco environment
#
#("/home/graspinglab/NearContactStudy/MDP/jaco/jaco.xml")
#
'''
Body ID 
2 = base
3 = link 1
4 = link 2
5 = link 3
6 = link 4
7 = link 5
8 = link 6
9 = link 7 
10 = finger 1 prox 
11 = finger 1 distal
12 = finger 2 prox
13 = finger 2 distal
14 = finger 3 prox
15 = finger 3 distal
16 = cube (object)
'''
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

		self._torque = [0,0,0,0,0,0,0,0,0,0]
		self._velocity = [0,0,0,0,0,0,0,0,0,0]

		self._jointAngle = [0,0,0,0,0,0,0,0,0,0]
		self._positions = [] # ??
		self._numSteps = 0
		self._simulator = "Mujoco"

		# define pid controllers for all joints
		# self.pid = [PID_(1.0,0.0,0.0), PID_(1.5,0.0,0.0), PID_(1.0,0.0,0.0),PID_(3,0.0,0.0), PID_(1.0,0.0,0.0), 
		# 	PID_(3.0,0.0,0.0),PID_(1.0,0.0,0.0), PID_(2.0,0.05,0.0), PID_(2.0,0.05,0.0), PID_(2.0,0.05,0.0)]
		# self.pid = [PID_(185,0.025,0.0), PID_(135,0.02,0.0), PID_(135,0.02,0.0)]
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

		# self.pid[3].set_target_jointAngle(thetas[3])
		# self.pid[4].set_target_jointAngle(thetas[4])
		# self.pid[5].set_target_jointAngle(thetas[5])
		# self.pid[6].set_target_jointAngle(thetas[6])
		# self.pid[7].set_target_jointAngle(thetas[7])
		# self.pid[8].set_target_jointAngle(thetas[8])
		# self.pid[9].set_target_jointAngle(thetas[9])


		# print("joint1",self.pid[1]._targetjA)

	def get_jointAngles(self, jA_ros):
		
		# joint 2 only because its origin is different from ros urdf
		if jA_ros[1] >= 3.14:
			jA_2 = jA_ros[1] - 3.14
		elif jA_ros[1] < 3.14:
			jA_2 = -(3.14 - jA_ros[1])

		return [(jA_ros[0]), jA_2, (jA_ros[2]+3.14), jA_ros[3], jA_ros[4], jA_ros[5], jA_ros[6], jA_ros[7], jA_ros[8], jA_ros[9]]


	def readfile(self, filename_pos, filename_vel):
		posfile = open(filename_pos,"r")
		velfile = open(filename_vel,"r")
		waypoints = []
		velocities = []
		for line in posfile.readlines():
			joint_states = line.rstrip('\n').split(',')
			for joint in range(len(joint_states)):
				try:
					# if joint == 0 or joint == 4 or joint == 6:
					# 	joint_states[joint] = float(joint_states[joint]) + 6.28
					# else:
					joint_states[joint] = float(joint_states[joint])
				except:
					pass
			waypoints.append(np.array(self.get_jointAngles(joint_states)))	

		for velline in velfile.readlines():
			joint_vels = velline.rstrip('\n').split(',')
			for vel in range(len(joint_vels)):
				try:
					joint_vels[vel] = float(joint_vels[vel])
				except:
					pass
			velocities.append(np.array(joint_vels[0:7]))			
		return waypoints, velocities

	# state representation : 3D poses of components in world coordinates
	def get_WorldCoord(self, bodyIDs):
		world_poses = []
		for i in range(len(bodyIDs)):
			world_poses.append(self._sim.data.body_xpos[bodyIDs[i]])
		return world_poses

	# state represetation : 3D poses of components wrt palm frame
	def get_PalmCoord(self, bodyIDs):
		palm_poses = []
		for j in range(len(bodyIDs)):
			temp_pose = np.array(self._sim.data.body_xpos[bodyIDs[i]])
			correspond_palm_pose = np.matmul(np.array(self._sim.data.body_xpos[10]), temp_pose)
			palm_poses.append(correspond_palm_pose)

		return palm_poses

	def get_metric(self, contact, f1_prox):
		if (contact.geom1 == "cube" and contact.geom2 == "f1_prox"):
			f1_prox_metric = contact.dist
		if (contact.geom1 == "cube" and contact.geom2 == "f2_prox"):
			f2_prox_metric = contact.dist
		if (contact.geom1 == "cube" and contact.geom2 == "f3_prox"):
			f3_prox_metric = contact.dist

		if (contact.geom1 == "cube" and contact.geom2 == "f1_dist"):
			f1_dist_metric = contact.dist
		if (contact.geom1 == "cube" and contact.geom2 == "f2_dist"):
			f2_dist_metric = contact.dist
		if (contact.geom1 == "cube" and contact.geom2 == "f3_dist"):
			f3_dist_metric = contact.dist

	def fwdkinematics(self):
		#### Forward kinematics ####
		for step in range(int(self._numSteps)):
			for i in range(7):
				self._sim.data.qpos[i] = theta_home[i]

			if max(np.absolute(self._sim.data.qpos[0:7]- theta_target[0:7])) < 0.001:
				if current == len(waypoints):
					# print("here")
					pass
					# break
				else:
					# print(waypoints[current])
					print ("current",self._sim.data.qpos[0:7])
					print ("target",theta_target[0:7])
					theta_target = waypoints[current]
					current += 1 

			else:
				# print ("current",self._sim.data.qpos[0:7])
				# print ("target",theta_target[0:7])
				for i in range(7):
					if theta_home[i] < theta_target[i]:
						theta_home[i] += 0.001
					elif theta_home[i] > theta_target[i]:
						theta_home[i] -= 0.001


			self._sim.forward()
			self._viewer.render()		

	def physim_mj(self):
		self.set_step(50)

		# theta_0 = np[0.0,0.0,0.0,3.14,0.0,3.14,0.0,0.0,0.0,0.0] # zero position, pointing up
		theta_0 = self.get_jointAngles([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.0, 0.0, 0.0, 0.0])

		# theta_home = [(4.71 - 3.14), (2.8 - 3.14), (3.14 - 0.1), 0.75, 4.62, 4.48, 4.88, 0.0, 0.0, 0.0] # home position 
		# theta_home = self.get_jointAngles([4.71, 2.8, 0.1, 0.75, 4.62, 4.48, 4.88, 0.0, 0.0, 0.0])
		theta_home = self.get_jointAngles([-1.5731004548310388, 2.8399635722906793, -3.528502071276302e-06, 0.7500470318119973, 4.619968819952383, 4.480029520452209, -1.4031059367326641, 0.0, 0.0, 0.0])

		pose = self.get_jointAngles([5.756481938389754, 5.364944086676359, 1.8223024418126335, 5.293841512048439, 6.879412325463892, 4.281757073790844, 4.897938695514569, 0.0, 0.0, 0.0])
		vel_origin = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]

		self.set_target_thetas(theta_0[7:]) # set joint angles for fingers
		
		self.data_all = []
		data1 = []
		data2 = []
		data3 = []
		data4 = []
		data5 = []
		data6 = []
		data7 = []


		
		filename_pos = "trajpos.csv"
		filename_vel = "trajvel.csv"
		filelift = "trajpos5.csv"
		fileliftvel = "trajvel5.csv"
		waypoints, velocities = self.readfile(filename_pos, filename_vel)
		liftpoints, lift_vels = self.readfile(filelift, fileliftvel)
		current = 0
	

		self._sim.data.qpos[0:10] = waypoints[0]
		# print ("before start", self._sim.data.qpos[0:10])
		theta_home = np.array(waypoints[0][:])		
		theta_target = theta_home[:]

		self._sim.data.qvel[0:7] = velocities[0]
		vel_target = velocities[0]
		


		##########################


		gripper = np.array([1.8, 1.8, 1.8])
		lift_count = 0
		count = 0
		pose = 0
		while True:
			# print("target:", theta_target[0:7])
			# print("actual:", self._sim.data.qpos[0:7])
			# print("current:",current)
			# print("x_pos", self._sim.data.body_xpos)

			#self._viewer.add_marker(pos=np.array([self._sim.data.body_xpos[10][0], self._sim.data.body_xpos[10][1], self._sim.data.body_xpos[10][2]]), size=np.array([0.02, 0.02, 0.02]))


			for i in range(3):
				self._jointAngle[i] = self._sim.data.sensordata[i+7]
				self._torque[i] = self.pid[i].get_Torque(self._jointAngle[i])
				self._sim.data.ctrl[i+7] = self._torque[i]

			# print(list(np.absolute(np.array(self._sim.data.qpos[0:7]) - theta_target[0:7])).index(max(list(np.absolute(np.array(self._sim.data.qpos[0:7]) - theta_target[0:7])))))
			# print((np.absolute(np.array(self._sim.data.sensordata[0:7]) - theta_target[0:7])))
			diff = list(np.absolute(np.array(self._sim.data.sensordata[0:7]) - theta_target[0:7]))

			# for k in range(len(waypoints)):
			# 	theta_target = waypoints[k]
			# 	# count =0 
			# print("pose:", pose)

			# print(self._sim.data.sensordata[10])

			# Motion control - moving to specific joint angles
			for j in range(7):
				self._sim.data.ctrl[j] = theta_target[j]
				self._sim.data.ctrl[10+j] = vel_target[j]
					# count += 1


			if max(np.absolute(np.array(self._sim.data.sensordata[0:7]) - theta_target[0:7])) <= 0.1:
				if current ==len(waypoints):
					# print("count:", count)
					count += 1

					if count > 10000: # after 10000 steps
						self.set_target_thetas(gripper)
						# for i in range(3):
						# 	self._jointAngle[i] = self._sim.data.sensordata[i+7]
						# 	self._torque[i] = self.pid[i].get_Torque(self._jointAngle[i])
						# 	self._sim.data.ctrl[i+7] = self._torque[i]
							# if gripper[i] < 0.8:
							# 	gripper[i] += 0.001
						# print("contacts",len(self._sim.data.contact))					
						# print("7: ", self._sim.data.sensordata[7])
						# print("8: ", self._sim.data.sensordata[8])					
						# print("9: ", self._sim.data.sensordata[9])					
					if count == 20000:
						theta_target = liftpoints[0]
						# lift_count +=1
						pose = 1
						current =0
					# pass
				else:
					# print("current", current)
					if pose == 0:
						theta_target = waypoints[current]
						vel_target = velocities[current]
						current += 1
					if pose == 1:
						# print("here")
						if lift_count < len(liftpoints):							
							theta_target = liftpoints[lift_count]
							vel_target = lift_vels[lift_count]
							lift_count += 1


			# else:
			# 	index_max = diff.index(max(diff))

			# 	for i in range(7):
			# 		self._sim.data.ctrl[10+i] = vel_target[i] 

			self._sim.step()
			self._viewer.render()


	def finger_control(self):
		for i in range(3):
			self._jointAngle[i+1] = self._sim.data.sensordata[i+1]
			self._torque[i+1] = self.pid[i+1].get_Torque(self._jointAngle[i+1])
			self._sim.data.ctrl[i+1] = self._torque[i+1]

	def wrist_control(self):
		self._jointAngle[0] = self._sim.data.sensordata[0]
		self._torque[0] = self.pid[0].get_Torque(self._jointAngle[0])
		self._sim.data.ctrl[0] = self._torque[0]


	# get 3D transformation matrix of each joint
	def get_joint_pose(self, joint_geom_name):
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
	def get_finger_pose(self, local_or_global):
		palm = self.get_joint_pose(["palm"])[0]
		# print(palm)
		finger_joints = self.get_joint_pose(["f1_prox", "f1_dist", "f2_prox", "f2_dist", "f3_prox", "f3_dist"])

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
	def get_rangefinder_data(self):
		range_data = []
		for i in range(9):
			range_data.append(self._sim.data.sensordata[i+4])

		return np.array(range_data)

	def sim_end_effector(self, action):
		initial_handpose = np.zeros(7)
		self._sim.data.qpos[0:7] = initial_handpose 
		initial_fingerpose = np.array([0.0, 0.0, 0.0, 0.0])
		# print(len(initial_fingerpose))
		gripper = np.array([0.0, 0.0, 0.0, 0.0])
		self.set_target_thetas(initial_fingerpose)
		step = 0
		start = time.time()


		while True:
			# print(self._sim.data.sensordata[:])
			# print("f1p:",self._sim.data.get_geom_xpos("f2_prox"))
			# print("f1d:",self._sim.data.get_geom_xpos("f2_dist"))
			# print(np.append(self._sim.data.get_geom_xmat("f1_prox")[0], self._sim.data.get_geom_xpos("f1_prox")[0]))
			pose = self.get_finger_pose("global")
			# print("here:",self._sim.data.get_site_xpos("f1_prox")[:])
			print("here:",self._sim.data.sensordata[5],self._sim.data.sensordata[7], self._sim.data.sensordata[9])

			# self._viewer.add_marker(pos=np.array([self._sim.data.get_site_xpos("f1_dist1")[0], self._sim.data.get_site_xpos("f1_dist1")[1], self._sim.data.get_site_xpos("f1_dist1")[2]]), size=np.array([0.002, 0.002, 0.002]))
			# self._viewer.add_marker(pos=np.array([self._sim.data.get_site_xpos("f1_dist2")[0], self._sim.data.get_site_xpos("f1_dist2")[1], self._sim.data.get_site_xpos("f1_dist2")[2]]), size=np.array([0.006, 0.006, 0.006]))

			self.wrist_control()
			self.finger_control()
			if step > 1000:				
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

			step += 1
			self._sim.step()
			self._viewer.render()



if __name__ == '__main__':
	print(os.path.dirname(os.path.realpath(__file__)))
	sim = Kinova_MJ("hand")
	# data = sim.physim_mj()
	sim.sim_end_effector([0.8, 0.8, 0.8])
	# plt.plot(data[0], 'r',data[1], 'g')
	# plt.show()