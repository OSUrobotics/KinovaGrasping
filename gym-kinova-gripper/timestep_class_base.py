#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:06:46 2021

@author: orochi
"""
import os
import time
import csv
from copy import deepcopy
from collections import OrderedDict
import json


class RecordTimestepBase():

    def __init__(self, phase):
        """ Timestep class contains the state, action, reward and time of a
        moment in the simulator. It contains this data as their respective
        classes but has methods to return the data contained in them and save
        the data to a csv
        @param phase - Phase class containing the state, action, reward as
        State, Action and Reward classes and the timestep and sim time as int
        and float"""
        self.state = deepcopy(phase.state)
        self.action = deepcopy(phase.action)
        self.reward = deepcopy(phase.reward)
        self.times = {'wall_time': time.time(), 'sim_time': deepcopy(phase.sim_time),
                      'timestep': deepcopy(phase.timestep)}

    def get_state_as_arr(self):
        """Method to return the state data as a single list
        @return - list of state values"""
        return self.state.get_obs()

    def get_action_as_arr(self):
        """Method to return the action data as a single list
        @return - list of action values"""
        action_profile = self.action.get_action()
        return list(action_profile[-1])

    def get_reward_as_arr(self):
        """Method to return the reward data as a single list
        @return - list of reward values"""
        reward, _ = self.reward.get_reward()
        return [reward]

    def get_full_timestep(self):
        """Method to return all stored data as one dictionary of lists"""
        data = OrderedDict()
        data['state'] = self.get_state_as_arr()
        data['action'] = self.get_action_as_arr()
        data['reward'] = self.get_reward_as_arr()
        data.update(self.times)
        return data

    def save_timestep_as_csv(self, file_name=None, write_flag='w'):
        """Method to save the timestep in a csv file
        @param file_name - name of file
        @param write_flag - type of writing, defaults to write but can be set
        to 'a' to append to the file instead"""
        path = os.path.dirname(os.path.abspath(__file__))
        if file_name is None:
            file_name = path + '/data/' + 'tstep_' + str(self.times['timestep']) +\
            '_wall_time_' + time.strftime("%m-%d-%y %H:%M:%S", time.localtime(self.times['wall_time'])) + '.csv'
        else:
            file_name = path + '/data/' + file_name + '.csv'
        data = self.get_full_timestep()
        header = ['Wall time', 'Sim time', 'Timestep']
        for i in range(len(data['state'])):
            header.append('State_'+str(i))
        for i in range(len(data['action'])):
            header.append('Action_'+str(i))
        for i in range(len(data['reward'])):
            header.append('reward_'+str(i))
        with open(file_name, write_flag, newline='') as csv_file:
            time_writer = csv.writer(csv_file, delimiter=',', quotechar='|',
                                     quoting=csv.QUOTE_MINIMAL)
            if write_flag == 'w':
                time_writer.writerow(header)
            time_writer.writerow([data['wall_time']] + [data['sim_time']] +
                                 [data['timestep']] + data['state'] +
                                 data['action'] + data['reward'])

    def save_timestep_as_json(self, file_name=None):
        """Method to save the timestep in a json file
        @param file_name - name of file"""
        path = os.path.dirname(os.path.abspath(__file__))
        if file_name is None:
            file_name = path + '/data/' + 'tstep_' + str(self.times['timestep']) +\
            '_wall_time_' + time.strftime("%m-%d-%y %H:%M:%S", time.localtime(self.times['wall_time'])) + '.json'
        else:
            file_name = path + '/data/' + file_name + '.json'
        _, reward_data = self.reward.get_reward()
        with open(file_name, 'w') as json_file:
            json.dump(self.times, json_file)
            json.dump(self.state.get_data_dict(), json_file)
            json.dump(self.action.get_action_dict(), json_file)
            json.dump(reward_data, json_file)
