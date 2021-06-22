#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:15:51 2021

@author: orochi
"""
from mujoco_py import MjSim

class Distance():
    def __init__(self,sim):
        print('initialized distance')
        self._sim=sim
        self.data={}        
    def get_metric(self, *keys):
        return self.data[keys]
        
    def update(self):
        #this looks into the simulator to find the data we are looking for
        self.data=[1]
        print('things')