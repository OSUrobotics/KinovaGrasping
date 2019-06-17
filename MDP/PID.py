# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:39:38 2018

@author: jack
"""
import csv

class PID(object):
    def __init__(self):
        self._kp = 0.1
        self._ki = 0.0
        self._kd = 0.0
        
        
        self._target_theta = 0.0
        self._sampling_time = 0.01
        
        self._theta0 = 0.0
        self._thetai = 0.0

    def init_status(self):
        self._theta0 = 0.0
        self._thetai = 0.0
    
    def set_target_theta(self, theta):
        self._target_theta = theta
        
    def get_target_theta(self):
        return self._target_theta
    
    def get_velocity(self, theta):
        error = self._target_theta - theta
        self._thetai += error * self._sampling_time
        dtheta = (error - self._theta0) / self._sampling_time
        self._theta0 = error
        
        
        duty_ratio = (error * self._kp + self._thetai * self._ki + dtheta * self._kd)/self._sampling_time
        
        if duty_ratio > 30:
            duty_ratio = 30
        elif duty_ratio < -30:
            duty_ratio = -30
        
        return duty_ratio