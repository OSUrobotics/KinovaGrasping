#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Learn finger control strategy 
# Summer 2019

###############

import os, sys
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy # actor-critic 
from stable_baselines.results_plotter import load_results
from kinova_gripper_env import *

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # to import cv2 successfully at stable_baselines



