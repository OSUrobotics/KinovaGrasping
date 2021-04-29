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
    valid_state_names={'Position':Position, 'Distance':Distance, 'Angle':Angle, 'Ratio':Ratio, 'Vector':Vector, 'Dot_Product':Dot_Product,'Group':State_Group}
    _sim=None
    def __init__(self,path='state.json'):
        with open(path) as f:
            self.data=json.load(f)
        for name,value in self.data.items():
            for key in State_Space.valid_state_names.keys():
                if key in name:
                    self.data[name]=State_Space.valid_state_names[key]({name:value})
                    
    def get_full_arr(self):
        arr=[]
        for name,value in self.data.items():
            temp=value.get_value()
            print('contents of', name, len(temp),temp)
            arr.extend(temp)
        return arr
        
    def get_value(self,keys):
        if type(keys) is str:
            keys=[keys]
        if len(keys)>1:
            data=self.data[keys[0]].get_specific(keys[1:])
        else:
            data=self.data[keys[0]].get_value()
        return data

    def update(self):
        for name,value in self.data.items():
            #print('updating ',name)
            self.data[name].update(name)

a=State_Space()
