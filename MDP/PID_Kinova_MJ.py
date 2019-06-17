#!/usr/bin/env python3

'''
Author : Yi Herng Ong
Purpose: PID control script for running simulation in Kinova
'''

import os, sys

'''
Input: joint position
Output: Torque
'''
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