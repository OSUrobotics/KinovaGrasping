#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:06:46 2021

@author: orochi
"""
import os
import time
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
        self.times = {'wall_time': time.time(), 'sim_time': sim_time,
                      'timestep': timestep}

    def get_state(self):
        """Method to return the state data"""
        return self.state.get_obs()

    def get_action(self):
        """Method to return the action data"""
        return self.action.get_action()

    def get_reward(self):
        """Method to return the reward data"""
        return self.reward.get_reward()

    def get_full_timestep(self):
        """Method to return all stored data as one dictionary"""
        data = {}
        data['state'] = self.state.get_obs()
        data['action'] = self.action.get_action()
        data['reward'] = self.reward.get_reward()
        data.update(self.times)
        return data

    def set_state(self, new_state):
        """Method to set the state
        @param state - State class containing info from this timestep"""
        self.state = new_state

    def set_action(self, new_action):
        """Method to set the action
        @param acton - Action class containing info from this timestep"""
        self.action = new_action

    def set_reward(self, new_reward):
        """Method to set the reward
        @param reward - Reward class containing info from this timestep"""
        self.reward = new_reward

    def set_time(self, timestep, sim_time, wall_time=None):
        """Method to set the time
        @param timestep - int represting the current timestep
        @param sim_time - float representing seconds since sim started in
        simulator
        @param wall_time - float represeting current computer time"""
        if wall_time is None:
            self.times = {'wall_time': time.time(), 'sim_time': sim_time,
                          'timestep': timestep}
        else:
            self.times = {'wall_time': wall_time, 'sim_time': sim_time,
                          'timestep': timestep}

    def save_timestep(self, file_name=None, write_flag='w'):
        """Method to save the timestep in a csv file
        @param file_name - name of file
        @param write_flag - type of writing, defaults to write but can be set
        to 'a' to append to the file instead"""
        path = os.path.dirname(os.path.abspath(__file__))
        if file_name is None:
            file_name = path + '/data/' + 'tstep_' + self.times['timestep'] +\
            '_wall_time_' + time.strftime("%m-%d-%y %H:%M:%S", time.localtime(self.times['wall_time'])) + '.csv'
        else:
            file_name = path + '/data/' + file_name
        data = self.get_full_timestep()
        with open(file_name, write_flag, newline='') as csv_file:
            time_writer = csv.writer(csv_file, delimiter=',', quotechar='|',
                                     quoting=csv.QUOTE_MINIMAL)
            time_writer.writerow(['Wall time'] + [data['wall_time']] +
                                 ['Sim time'] + [data['sim_time']] +
                                 ['Timestep'] + [data['timestep']])
            time_writer.writerow(['State'] + data['state'])
            time_writer.writerow(['Action'] + data['action'])
            time_writer.writerow(['reward'] + data['reward'])
