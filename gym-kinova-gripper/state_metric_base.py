#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 13:59:44 2021

@author: orochi
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_classes.stats_tracker_base import *
from collections import OrderedDict


class StateMetricBase:
    _sim = None

    def __init__(self, data_structure):
        print('intializing with data strucure:', data_structure)
        try:
            self.data = StatsTrackerArray(data_structure[0], data_structure[1])
            print('stats tracker array initialized with min and max',
                  self.data.allowable_min, self.data.allowable_max)
        except TypeError:
            self.data = StatsTrackerBase(data_structure[0], data_structure[1])
            print('stats tracker base initialized with min and max',
                  self.data.allowable_min, self.data.allowable_max)
        except KeyError:
            self.data = []

    def get_value(self):
        return self.data.value

    def update(self):
        None

class StateMetricAngleBase(StateMetricBase):
    def update(self, keys):
        None


class StateMetricPositionBase(StateMetricBase):
    def update(self, keys):
        None


class StateMetricVectorBase(StateMetricBase):
    def update(self, keys):
        None


class StateMetricRatioBase(StateMetricBase):
    def update(self, keys):
        None


class StateMetricDistanceBase(StateMetricBase):
    def update(self, keys):
        None


class StateMetricDotProductBase(StateMetricBase):
    def update(self, keys):
        None
