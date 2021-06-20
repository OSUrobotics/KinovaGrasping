#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:56:31 2021

@author: orochi
"""
import json
import numpy as np


class Action():
    _sim = None

    def __init__(self, desired_speed, json_path='action.json'):
        """ Desired speed is the new speed, we assume the initial speed is 0
        @param desired_speed - list of speeds with length described in action.json
        @param json_path - path to json file"""
        # these parameters can be pulled from a file created when the
        # simulator starts
        self.time = 0.002  # length of a timestep in seconds
        self.timesteps = 15  # number of simulation timesteps per step call
        self.max_acceleration = 20  # rad/s^2
        self.min_acceleration = 0.2  # rad/s^2
        self.action_profile = []

        with open(json_path) as f:
            self.old_speed = json.load(f)

        if type(desired_speed) is dict:
            self.new_speed = desired_speed
        else:
            self.new_speed = self.old_speed.copy()
            self.set_speed(desired_speed, 'old')

        self.build_action()

    def get_full_arr(self, flag='new'):
        """Returns the data as a list instead of a dict"""
        if flag == 'old':
            return self.collect_data(self.old_speed)
        elif flag == 'new':
            return self.collect_data(self.new_speed)
        else:
            print('flag should be "old" or "new".')

    def get_action(self):
        """Returns the speeds to get from old speed to new speed as a list of lists"""
        return self.action_profile

    def collect_data(self, data):
        """iterates through data to return all its values as list, even if data is dict of dicts
        @param data - a dict"""
        data_arr = []
        for name, value in data.items():
            if type(value) is dict:
                data_arr.extend(self.collect_data(value))
            elif type(value) is list:
                data_arr.extend(value)
            else:
                data_arr.append(value)
        return data_arr

    def build_action(self):
        """Builds the action profile (speed profile to get from old speed to new speed)"""
        speed = self.get_full_arr('old')
        ending_speed = self.get_full_arr('new')
        direction = [np.sign(ending_speed[i]-speed[i]) for i in
                     range(len(speed))]
        self.action_profile = np.zeros([self.timesteps, len(speed)])
        for i in range(len(speed)):
            if abs(ending_speed[i] - speed[i]) / (self.time*self.timesteps) >\
                                                        self.max_acceleration:
                print('Caution! Desired speed is too different from current\
                      speed to reach in', self.timesteps, 'timesteps.\
                      Action will apply max acceleration for all steps but \
                      this will not reach desired speed!')
        for i in range(len(speed)):
            if direction[i] > 0:
                self.action_profile[0][i] = min(speed[i]+self.max_acceleration
                                                * self.time, ending_speed[i])
            else:
                self.action_profile[0][i] = max(speed[i]-self.max_acceleration
                                                * self.time, ending_speed[i])
        for j in range(self.timesteps-1):
            for i in range(len(speed)):
                if direction[i] > 0:
                    self.action_profile[j+1][i] = min(self.action_profile[j][i]
                                                      + self.max_acceleration *
                                                      self.time,
                                                      ending_speed[i])
                else:
                    self.action_profile[j+1][i] = max(self.action_profile[j][i]
                                                      - self.max_acceleration *
                                                      self.time,
                                                      ending_speed[i])
        return self.action_profile

    def set_speed(self, speed, flag='new'):
        """sets speed speed and calculates the new action profile"""
        if len(speed) != len(self.old_speed):
            print('desired speed is not the same length as current speed,\
                  speed not set. Speed should have length',
                  len(self.old_speed))
            return
        if flag == 'old':
            for i, name in enumerate(self.old_speed.keys()):
                self.old_speed[name] = speed[i]
        elif flag == 'new':
            for i, name in enumerate(self.old_speed.keys()):
                self.old_speed[name] = self.new_speed[name]
                self.new_speed[name] = speed[i]
        else:
            print('flag should be "old" or "new".')
        return self.build_action()


if __name__ == "__main__":
    a = Action([0, 0, 0, 0, 0, 0])
    a.set_speed([0, 0, 0, -0.3, 0.3, 0.3])
    print(a.build_action())
