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
from collections import OrderedDict


class StateSpace:
    valid_state_names = {'Position': Position, 'Distance': Distance, 'Angle': Angle, 'Ratio': Ratio, 'Vector': Vector,
                         'DotProduct': DotProduct, 'StateGroup': StateGroup}
    _sim = None

    def __init__(self, path='state.json'):
        with open(path) as f:
            json_data = json.load(f)
        self.data = OrderedDict()
        for name, value in json_data.items():
            state_name = name.split(sep='_')
            try:
                self.data[name] = eval(state_name[0] + '(value)')
            except NameError:
                print(state_name[0],'ya done messed up a-a-ron, valid state names are', [name for name in
                                                                           StateSpace.valid_state_names.keys()])

    def get_full_arr(self):
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


if __name__ == "__main__":
    a = StateSpace()
    print(a.get_full_arr())
