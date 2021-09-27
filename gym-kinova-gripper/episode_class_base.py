#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:16:32 2021

@author: orochi
"""
from collections import OrderedDict
from timestep_class_base import TimestepBase


class EpisodeBase():
    """ Episode class contains the timesteps from a single episode, methods to
    acces the data and a method to save it in a csv file
    @param shape - shape being picked up
    @param size - size of shape being grabbed
    @param orientation - orientation of grasp
    @param episode_num - current episode if running multiple"""
    def __init__(self, shape='', size='', orientation='', episode_num=0):
        self.data = OrderedDict()
        self.episode_num = episode_num
        self.shape = shape
        self.size = size
        self.orientation = orientation

    def get_timestep(self, timestep):
        """Method to return the data in the timestep associated with the string
        or int given
        @param timestep - int or string of desired timestep"""
        try:
            data = self.data[timestep]
        except KeyError:
            key = 't_'+str(timestep)
            data = self.data[key]
        return data.get_full_timestep()

    def get_full_episode(self):
        """Method to return the data in all the timesteps in an ordered dict"""
        data = OrderedDict()
        for k, v in self.data.items():
            data[k] = v.get_full_timestep()
        return data

    def save_episode(self, file_name=None):
        """Method to save the episode
        @param file_name - name of file"""
        flag = True
        if file_name is None:
            file_name = 'episode_' + str(self.episode_num) + self.shape +\
                        self.size + self.orientation + '.csv'
        for i in self.data.values():
            if flag:
                i.save_timestep(file_name=file_name, write_flag='w')
                flag = False
            else:
                i.save_timestep(file_name=file_name, write_flag='a')

    def add_timestep(self, timestep):
        """Method to add a new timestep to the episode
        @param timestep - Timestep class containing all relevant data"""
        self.data['t_' + str(timestep.times['timestep'])] = timestep


if __name__ == '__main__':
    a = EpisodeBase(shape='fake_cube', episode_num=1)
    state = [1, 2, 3]
    action = [23]
    reward = [155]
    timestep = 0
    sim_time = 0.02
    b = TimestepBase(state, action, reward, timestep, sim_time)
    a.add_timestep(b)
    b = TimestepBase(state, action, reward, 1, 0.04)
    a.add_timestep(b)
    b = TimestepBase(state, action, reward, 2, 0.06)
    a.add_timestep(b)
    b = TimestepBase(state, action, reward, 3, 0.08)
    a.add_timestep(b)
    a.save_episode()
