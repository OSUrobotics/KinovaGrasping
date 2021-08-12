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
from state_metric_base import StateMetricBase
from collections import OrderedDict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_classes.stats_tracker_base import *

class StateMetricGroup(StateMetricBase):
    valid_state_names = {'Position': StateMetricPosition, 'Distance': StateMetricDistance, 'Angle': StateMetricAngle, 'Ratio': StateMetricRatio, 'Vector': StateMetricVector,
                         'DotProduct': StateMetricDotProduct, 'StateGroup':'StateMetricGroup'}
    def __init__(self, data_structure):
        super().__init__(data_structure)
        self.data = OrderedDict()
        for name, value in data_structure.items():
            state_name = name.split('_')
            try:
                print('state name',state_name[0])
                self.data[name] = StateMetricGroup.valid_state_names[state_name[0]](value)
            except TypeError:
                self.data[name] = StateMetricGroup(value)
            except KeyError:
                print('Invalid state name. Valid state names are', [name
                          for name in StateMetricGroup.valid_state_names.keys()])

    def update(self, keys):
        arr = []
        for name, value in self.data.items():
            temp = value.update(keys + '_' + name)
            arr.append(temp)
        return self.data

    def search_dict(self, subdict, arr=[]):
        for name, value in subdict.items():
            if type(value) is dict:
                arr = self.search_dict(subdict[name], arr)
            else:
                try:
                    arr.extend(value.get_value())
                except TypeError:
                    arr.extend([value.get_value()])
        return arr

    def get_value(self):
        return self.search_dict(self.data, [])

class StateSpaceBase():
    valid_state_names = {'Position': StateMetricPosition, 'Distance': StateMetricDistance, 'Angle': StateMetricAngle, 'Ratio': StateMetricRatio, 'Vector': StateMetricVector,
                         'DotProduct': StateMetricDotProduct, 'StateGroup': StateMetricGroup}
    _sim = None

    def __init__(self, path=os.path.dirname(__file__)+'/config/state.json'):
        with open(path) as f:
            json_data = json.load(f)
        self.data = OrderedDict()
        for name, value in json_data.items():
            state_name = name.split(sep='_')
            try:
                self.data[name] = StateSpaceBase.valid_state_names[state_name[0]](value)
            except NameError:
                print(state_name[0],'Invalid state name. Valid state names are', [name for name in
                                                                           StateSpaceBase.valid_state_names.keys()])

    def get_obs(self):
        #self.update()
        arr = []
        for name, value in self.data.items():
            temp = value.get_value()
            try:
                arr.extend(temp)
            except TypeError:
                arr.extend([temp])
        return arr

    def get_value(self, keys):
        if type(keys) is str:
            keys = [keys]
        if len(keys) > 1:
            data = self.data[keys[0]].get_specific(keys[1:])
        else:
            data = self.data[keys[0]].get_value()
        return data

    def update(self):
        for name, value in self.data.items():
            self.data[name].update(name)
        return self.get_obs()



if __name__ == "__main__":
    a = StateSpaceBase()
    print(a.get_obs())
