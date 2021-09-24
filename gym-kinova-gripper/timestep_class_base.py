#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:06:46 2021

@author: orochi
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_classes.stats_tracker_base import StatsTrackerArray
from collections import OrderedDict
import state_space as sim_state
import action_class_base as sim_action
import reward_base as sim_reward
import csv

class TimestepBase():

    def __init__(self, state, action, reward, timestep, sim_time):
        """ Timestep class contains the state, action, reward and time of a
        moment in the simulator. It contains this data as their respective
        classes but has methods to return the data contained in them and save
        the data to a csv
        @param state - State class containing the state of the current timestep
        @param action - Action class containing action of current timestep
        @param reward - Reward class containing reward of current timestep
        @param timestep - current step in simulator. Starts at 0
        @param sim_time - time in seconds of the simulator"""
        self.state = state
        self.action = action
        self.reward = reward
        self.times = {'wall_time': time.time(), 'sim_time': sim_time, 'timestep': timestep}

    def get_state(self):
        return self.state.get_obs()

    def get_action(self):
        return self.action.get_action()

    def get_reward(self):
        return self.reward.get_reward()

    def get_full_timestep(self):
        data = {}
        data['state'] = self.state.get_obs()
        data['action'] = self.action.get_action()
        data['reward'] = self.reward.get_reward()
        data.update(self.times)
        return data

    def set_state(self, new_state):
        self.state = new_state

    def set_action(self, new_action):
        self.action = new_action

    def set_reward(self, new_reward):
        self.reward = new_reward

    def set_time(self, timestep, sim_time, wall_time=None):
        if wall_time is None:
            self.times = {'wall_time': time.time(), 'sim_time': sim_time, 'timestep': timestep}
        else:
            self.times = {'wall_time': wall_time, 'sim_time': sim_time, 'timestep': timestep}

    def save_timestep(self):
        path = os.path.abspath(__file__)
        file_name = path + '/data/' + 'tstep_' + self.times['timestep'] + '_wall_time_' + time.strftime("%m-%d-%y %H:%M:%S", time.localtime(self.times['wall_time'])) + '.csv'
        data = self.get_full_timestep()
        with open(file_name,'w', newline='') as csv_file:
            time_writer = csv.writer(csv_file, delimiter=',', quotechar='|', 
                                     quoting=csv.QUOTE_MINIMAL)
            time_writer.writerow(['Wall time'] + [data['wall_time']] + ['Sim time'] + [data['sim_time']] + ['Timestep'] + [data['timestep']])
            time_writer.writerow(['State'] + data['state'])
            time_writer.writerow(['Action'] + data['action'])
            time_writer.writerow(['reward'] + data['reward'])