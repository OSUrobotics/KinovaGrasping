#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:56:40 2021

@author: orochi
"""
import pickle
import numpy as np
import sys
sys.path.insert(0, '/home/orochi/redownload/KinovaGrasping/steph branch')
from core_classes.stats_tracker_base import *


class Reward():

    _sim = None

    def __init__(self, lift_scale=50, grasp_scale=0.0, finger_scale=0.0):
        """ scaleing ratio of each term determines its relative importance
        @param lift_scale - single number
        @param grasp_scale - single number
        @param finger_scale - single number"""
        self.Grasp_net = pickle.load(open(self.file_dir +'/kinova_description/gc_model.pkl', "rb"))
        self.Grasp_reward = False

        self.heights = StatsTrackerBase(-0.005,0.3)

        self.finger_scale = finger_scale
        self.grasp_scale = grasp_scale
        self.lift_scale = lift_scale

    def get_reward(self, state):
        obj_pose = Reward._sim.data.get_geom_xpos('object')
        self.heights.set_value(obj_pose[-1])
        finger_joints = ["f1_prox", "f1_prox_1", "f2_prox", "f2_prox_1",
                         "f3_prox", "f3_prox_1", "f1_dist", "f1_dist_1",
                         "f2_dist", "f2_dist_1", "f3_dist", "f3_dist_1"]

        # Finger reward 
        finger_obj_dists = []
        for i in finger_joints:
            pos = self._sim.data.get_site_xpos(i)
            dist = np.absolute(pos[0:3] - obj_pose[0:3])
            temp = np.linalg.norm(dist)
            finger_obj_dists.append(temp)

        finger_reward = -np.sum((np.array(finger_obj_dists[:6])) + (np.array(finger_obj_dists[6:])))

        # Grasp reward
        grasp_quality = self.Grasp_net.predict(np.array(state[0:75]).reshape(1,-1))
        if (grasp_quality >= 0.3) & (not self.Grasp_Reward):
            grasp_reward = self.grasp_scale
            self.Grasp_Reward = True
        else:
            grasp_reward = 0.0

        # Lift reward
        lift_reward = (self.heights.value - self.heights.min_found)/0.2*self.lift_scale

        finger_reward = 0
        reward = finger_reward + lift_reward + grasp_reward

        info = {"finger_reward": finger_reward, "grasp_reward": grasp_reward,
                "lift_reward": lift_reward}

        return reward, info


if __name__ == "__main__":
    r = Reward()
    r.get_reward()
