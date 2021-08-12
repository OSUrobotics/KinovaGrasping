#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:33:29 2021

@author: orochi
"""
import os

class TimeParamBase():
    def __init__(self, json_path = os.path.dirname(__file__)+'/config/time.json'):
        with open(path) as f:
            json_data = json.load(f)
        self.timestep_len = json_data['timestep_len']
        self.timestep_num = json_data['timestep_num']
        try:
            max_sim_time = json_data['max_sim_time']
            self.max_steps = int(max_sim_time / (self.timestep_len * self.timestep_num))
        except KeyError:
            self.max_steps = None
    
    def write_params(self,filepath):
        """this function writes the timestep parameters to the apropriate file
        to update the simulator"""
        None