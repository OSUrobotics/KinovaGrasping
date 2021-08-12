#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:56:31 2021

@author: orochi
"""
import json
import numpy as np
import os
import sys
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_classes.stats_tracker_base import StatsTrackerArray
from collections import OrderedDict
from action_class_base import ActionBase

class ActionMujoco(ActionBase):
    _sim = None

    def __init__(self, starting_speed=None, acceleration_range=[0.2, 20],
                 json_path=os.path.dirname(__file__)+'/config/action.json'):
        """ Starting speed is the speed of the fingers at initialization. We
        assume the initial speed is 0 if not otherwise specified.
        @param desired_speed - list of speeds with length described in
        action.json
        @param json_path - path to json file"""
        super().__init__(starting_speed, acceleration_range, json_path)
        self.starting_coords = []
        self.wrist_pose = []

    def build_action(self):
        """Builds the action profile (speed profile to get from old speed to
        new speed)"""
        speed = np.array(self.last_speed.value)
        ending_speed = np.array(self.current_speed.value)
        direction = [np.sign(ending_speed[i]-speed[i]) for i in
                     range(len(speed))]
        action_profile = np.zeros([self.timesteps, len(speed)])
        if any(abs(ending_speed - speed) / (self.time*self.timesteps) >
               self.max_acceleration):
            warnings.warn('Desired speed is too different from current\
                  speed to reach in ' + str(self.timesteps) + ' timesteps.\
                  Action will apply max acceleration for all steps but \
                  this will not reach the desired speed!')
        for i in range(len(speed)):
            if direction[i] > 0:
                action_profile[0][i] = min(speed[i]+self.max_acceleration
                                           * self.time, ending_speed[i])
            else:
                action_profile[0][i] = max(speed[i]-self.max_acceleration
                                           * self.time, ending_speed[i])
        for j in range(self.timesteps-1):
            for i in range(len(speed)):
                if direction[i] > 0:
                    action_profile[j+1][i] = min(action_profile[j][i]
                                                 + self.max_acceleration *
                                                 self.time, ending_speed[i])
                else:
                    action_profile[j+1][i] = max(action_profile[j][i]
                                                 - self.max_acceleration *
                                                 self.time, ending_speed[i])
        return action_profile

    def immobilize_hand(self):
        """Adds in forces to the sliders to prevent the hand from moving due
        to gravity or collisions with the object."""
        error = self.starting_coords - self.wrist_pose
        kp = 50
        mass = 0.733
        gear = 25
        slider_motion = np.matmul(self.Tfw[0:3, 0:3], [kp * error[0], kp * error[1], kp * error[2] + mass * 10 / gear])
        slider_motion[0] = -slider_motion[0]
        slider_motion[1] = -slider_motion[1]
        return slider_motion


if __name__ == "__main__":
    a = Action()
    print(a.set_speed([0, 0, 0, 0.3, 0.3, 0.3]))
    print(a.set_speed([0.2, 0.5, 0.1, 0.5, 0.9, 0.0]))
