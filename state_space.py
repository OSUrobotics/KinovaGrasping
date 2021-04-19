#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:57:24 2021

@author: orochi
"""
import json
import time
import numpy as np
from state_metric import *

class State_Space:
    valid_state_names={'Position':Position,'Distance':Distance,'Angle':Angle,'Ratio':Ratio,'Vector':Vector, 'Group':State_Group}
    def __init__(self,sim,path='state.json'):
        self.sim=sim
        with open(path) as f:
            self.template=json.load(f)
        for name,value in self.template.items():
            print('looking at ',name,value)
            for key in State_Space.valid_state_names.keys():
                if key in name:
                    self.template[name]=State_Space.valid_state_names[key](self.sim,{name:value})
                    
    def get_full_arr(self):
        return self.collect_data(self.data)
        
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
        
    def set_full_arr(self,stuff):
        if len(stuff)!=self.len:
            print(f'length of data incorrect! length of the data you are setting should be {self.len}. data not saved.')
        else:
            self.set_data(self.data,stuff.copy())
    
    def set_data(self,data,new_value,keys=[]):
        for name,value in data.items():
            keys.append(name)
            if type(value) is dict:
                new_value=self.set_data(value,new_value,keys)
            elif type(value) is list:
                for i in range(len(value)):
                    keys.append(i)
                    self.set_value(keys,new_value[0])
                    keys.pop()
                    new_value.pop(0)
            else:
                self.set_value(keys,new_value[0])
                new_value.pop(0)
            keys.pop()
        return new_value
    
    def set_value(self,keys,value):
        data=self.data
        if type(keys) is str:
            keys=[keys]
        for key in keys[:-1]:
            try:
                data[key]
            except:
                print(f'Key name: "{key}" is not valid. Please check your inputs.')
                return
            data=data.setdefault(key,{})
        try:
            if np.shape(value)!=np.shape(data[keys[-1]]):
                print('Input value not the same length as current value. Please check your inputs.')
            else:
                data[keys[-1]]=value
        except:
            print(f'Key name: "{keys[-1]}" is not valid. Please check your inputs.')
    
    def learn_inherit(self):
        print('inheritance works this way')
        
    def get_value(self,keys):
        if type(keys) is str:
            keys=[keys]
        data=self.data
        for key in keys[:-1]:
            try:
                data[key]
            except:
                print(f'Key name: "{key}" is not valid. Please check your inputs.')
                return
            data=data.setdefault(key,{})
        return data[keys[-1]]



a=State_Space(1)
