#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:12:08 2021

@author: orochi
"""
from collections import OrderedDict
from timestep_class_base import RecordTimestepBase


class RecordEpisodeMujoco():
    valid_shapes = ['Cube', 'Cylinder', 'Hourglass', 'Vase1', 'Vase2']
    valid_sizes = ['B', 'M', 'S']
    valid_orientations = ['normal', 'top', 'rotated']

    def __init__(self, identifier):
        """ Episode class contains the timesteps from a single episode, methods
        to acces the data and a method to save it in a csv file
        @param identifier - str containing name of episode used for identifying
        datafiles in the futureshape being picked up"""
        self.data = OrderedDict()
        self.identifier = identifier

    @staticmethod
    def build_identifier(shape, size, orientation, episode_num):
        """ Static method to take in an arbitrary number of features and use
        them to build a unique identifier for the episode. Example features
        include episode number, shape being grabbed, hand used etc.
        @return identifier - str containing name of episode used for
        identifying datafiles in the futureshape being picked up"""
        if shape not in RecordEpisodeMujoco.valid_shapes:
            raise Exception('Shape not in list of valid_shapes, valid shapes \
                            are', RecordEpisodeMujoco.valid_shapes)
        elif size not in RecordEpisodeMujoco.valid_sizes:
            raise Exception('Size not in list of valid_sizes, valid sizes \
                            are', RecordEpisodeMujoco.valid_orientations)
        elif orientation not in RecordEpisodeMujoco.valid_shapes:
            raise Exception('Shape not in list of valid_orientations, valid \
                            orientations are', RecordEpisodeMujoco.valid_orientations)
        return 'episode_' + str(episode_num) + shape + size + '_' + orientation

    def add_timestep(self, timestep):
        """Method to add a new timestep to the episode
        @param timestep - RecordTimestep class containing all relevant data"""
        self.data['t_' + str(timestep.times['timestep'])] = timestep

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
            file_name = self.identifier + '.csv'
        for i in self.data.values():
            if flag:
                i.save_timestep_as_csv(file_name=file_name, write_flag='w')
                flag = False
            else:
                i.save_timestep_as_csv(file_name=file_name, write_flag='a')


if __name__ == '__main__':
    a = RecordEpisodeMujoco(shape='fake_cube', episode_num=1)
    state = [1, 2, 3]
    action = [23]
    reward = [155]
    timestep = 0
    sim_time = 0.02
    b = RecordTimestepBase(state, action, reward, timestep, sim_time)
    a.add_timestep(b)
    b = RecordTimestepBase(state, action, reward, 1, 0.04)
    a.add_timestep(b)
    b = RecordTimestepBase(state, action, reward, 2, 0.06)
    a.add_timestep(b)
    b = RecordTimestepBase(state, action, reward, 3, 0.08)
    b.save_timestep_as_json()
    a.add_timestep(b)
    a.save_episode()
