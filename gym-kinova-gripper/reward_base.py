#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:56:40 2021

@author: orochi
"""
import pickle
import numpy as np
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_classes.stats_tracker_base import *
import state_space
from collections import OrderedDict

class RewardBase():

    _sim = None

    def __init__(self,json_path=os.path.dirname(__file__)+'/config/reward.json'):
        """ """
        with open(json_path) as f:
            json_data = json.load(f)
        reward_params = json_data['Parameters'] # this is used to set any weights or paths in the individual finger/lift/grasp rewards
        self.reward_weights = json_data['Reward']

        self.reward = {}

    def get_reward(self):
        reward = 0
        for i in self.reward_weights.keys():
            self.reward[i] = eval('self.get_' + i + '_reward()') * self.reward_weights[i]
            reward += self.reward[i]
        return reward, self.reward

    def get_finger_reward(self):
        return 0

    def get_lift_reward(self):
        return 0

    def get_grasp_reward(self):
        return 0


if __name__ == "__main__":
    r = RewardBase()
    r.get_reward()
