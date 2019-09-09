#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: generate velocity using PID controller based on current object state
# Summer 2019
###############

'''
state representation : joint angles, world coordinates of each finger, object, object size, object shape etc...

'''

import os, sys
import numpy as np

def expertPID(_obs):
	obj_size = _obs[0]
	obj_shape = 