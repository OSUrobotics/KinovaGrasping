#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:56:31 2021

@author: orochi
"""
import json

class Action:
    _sim=None
    def __init__(self,curr_speed,path='action.json'):
        with open(path) as f:
            self.new_speed=json.load(f)
        if type(curr_speed) is dict:
            self.old_speed=curr_speed
        else:
            self.old_speed=self.new_speed.copy()
            self.set_speed(curr_speed,'old')
        self.len=len(self.get_full_arr())
        
        #these parameters can be pulled from a file created when the simulator starts
        self.time=0.0001
        self.timesteps=1
        self.max_acceleration=2
        self.min_acceleration=0.2
        self.steps=[]
        self.build_action()

    def get_full_arr(self,flag='new'):
        if flag=='old':
            return self.collect_data(self.old_speed)
        elif flag=='new':
            return self.collect_data(self.new_speed)
        else:
            print('flag should be "old" or "new".')
            
    def collect_data(self,data):
        data_arr=[]
        for name,value in data.items():
            if type(value) is dict:
                data_arr.extend(self.collect_data(value))
            elif type(value) is list:
                data_arr.extend(value)
            else:
                data_arr.append(value)
        return data_arr
    
    def build_action(self):
        self.steps=[self.old_speed.copy()]
        speed=self.get_full_arr('old')
        ending_speed=self.get_full_arr('new')
        for i in range(len(speed)):
            if abs(ending_speed[i]-speed[i])/(self.time*self.timesteps)>self.max_acceleration:
                print('Caution! Desired speed is too different from current speed to reach in 1 set of timesteps. Action will apply max acceleration for all steps but this will not reach desired speed!')
        for i in range(self.timesteps):
            next_speed=speed.copy()
            for j in range(len(next_speed)):
                pass
            
    def set_speed(self,speed,flag='new'):
        if flag=='old':
            for name,value in self.old_speed.items():
                self.old_speed[name]=speed[0]
                speed.pop(0)
        elif flag=='new':
            for name,value in self.new_speed.items():
                self.new_speed[name]=speed[0]
                speed.pop(0)
        else:
            print('flag should be "old" or "new".')
            
a=Action([1,2,3,4,5,6])
