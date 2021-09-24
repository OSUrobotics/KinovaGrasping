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


class ActionBase():
    _sim = None

    def __init__(self, starting_speed=None, acceleration_range=[0.2, 20],
                 json_path=os.path.dirname(__file__)+'/config/action.json'):
        """ Starting speed is the speed of the fingers at initialization. We
        assume the initial speed is 0 if not otherwise specified.
        @param desired_speed - list of speeds with length described in
        action.json
        @param json_path - path to json file"""
        self.min_acceleration = acceleration_range[0]  # rad/s^2
        self.max_acceleration = acceleration_range[1]  # rad/s^2
        self.action_profile = []

        with open(json_path) as f:
            json_conts = json.load(f)

        # Pull the important parameters from the json file
        # They will be changed to class varriables when we add a function to
        # autogenerate a json file with these parameters and others when the
        # simulator starts.
        params = json_conts['Parameters']
        self.time = params['Timestep_len']  # length of a timestep in seconds
        self.timesteps = params['Timestep_num']  # number of simulation
        #                                          timesteps per step call

        # Set up the current and last speed values with max and min speeds and
        # the order of actions pulled from the json file
        action_struct = json_conts['Action']
        self.action_order = list(action_struct.keys())
        action_min_and_max = np.array(list(action_struct.values()))
        self.current_speed = StatsTrackerArray(action_min_and_max[:, 0],
                                               action_min_and_max[:, 1])
        self.last_speed = StatsTrackerArray(action_min_and_max[:, 0],
                                            action_min_and_max[:, 1])
        # set initial values of the speed
        try:
            self.last_speed.set_value(starting_speed)
            self.current_speed.set_value(starting_speed)
        except TypeError:
            self.last_speed.set_value(np.zeros(len(action_min_and_max)))
            self.current_speed.set_value(np.zeros(len(action_min_and_max)))

    def get_action(self):
        """Returns the speeds to get from old speed to new speed as a list
        of lists"""
        return self.action_profile

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

    def set_speed(self, speed):
        """sets last speed to current speed's value and sets current speed to
        input speed value, then calculates the new action profile"""
        if len(speed) != len(self.last_speed.value):
            raise IndexError('desired speed is not the same length as current\
                             speed, speed not set. Speed should have length'
                             + str(len(self.old_speed)))
        self.last_speed.set_value(self.current_speed.value)
        self.current_speed.set_value(speed)
        self.action_profile = self.build_action()
        return self.action_profile

    def get_name_value(self):
        """sets speed speed and calculates the new action profile"""
        action_dict = OrderedDict()
        for i, j in zip(self.action_order, self.current_speed.value):
            action_dict[i] = j
        return action_dict


if __name__ == "__main__":
    a = ActionBase()
    print(a.set_speed([0, 0, 0, 0.3, 0.3, 0.3]))
    print(a.set_speed([0.2, 0.5, 0.1, 0.5, 0.9, 0.0]))
