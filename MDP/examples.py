# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:37:34 2018

@author: jack

"""

from mujoco_py import *
from PID import *
from Rotations import *
import time
import csv
import math

repeats = 20       
class MUJOCO(object):
    def __init__(self):
        self._pid = [PID(),PID(),PID(),PID(),PID(),PID(),PID(),PID(),PID()]
        self._linearVelocity = [0,0,0,0,0,0,0,0,0]
        self._theta = [0,0,0,0,0,0,0,0,0]
        self._positions =[]
        self._convertdeg2rad = 57.295779578552 
        self._num_steps = 0
        self._timestep = 0.0001
        
        self._simulator = "Mujoco"
        self._physics_engine = ""
        self._experiment = ""
        self._current_iteration = 0

        
        self._model = load_model_from_path("/home/graspinglab/NearContactStudy/MDP/kinova_description/m1n6s300.xml")
        self._sim = MjSim(self._model)
        self._viewer = MjViewer(self._sim)
        self._sim.model.opt.timestep = self._timestep
        
    def set_current_iteration(self, iteration):
        self._current_iteration= iteration
        
    def set_experiment(self,experiment):
        self._experiment = experiment
        
    def set_num_steps(self):
        self._num_steps = simSteps(self._experiment,self._timestep)
        
    def run_mujoco(self):
        self.set_num_steps()
        for simStep in range(self._num_steps):  
            
            self._pid = set_target_thetas(self._num_steps, self._pid,self._experiment,self._simulator,simStep)
                
            if simStep % 500 == 0:
                for jointNum in range(6):
                    # print(self._sim.data.sensordata[jointNum])
                    self._theta[jointNum] = self._sim.data.sensordata[jointNum]
                    self._linearVelocity[jointNum] = self._pid[jointNum].get_velocity(math.degrees(self._theta[jointNum]))/self._convertdeg2rad
                    self._sim.data.ctrl[jointNum] = self._linearVelocity[jointNum]
                    # print('velocity',self._linearVelocity[jointNum])
                self._positions.append([self._sim.data.get_body_xpos('m1n6s300_link_6')[0],self._sim.data.get_body_xpos('m1n6s300_link_6')[1],self._sim.data.get_body_xpos('m1n6s300_link_6')[2],self._sim.data.get_body_xquat('m1n6s300_link_6')[0],self._sim.data.get_body_xquat('m1n6s300_link_6')[1],self._sim.data.get_body_xquat('m1n6s300_link_6')[2],self._sim.data.get_body_xquat('m1n6s300_link_6')[3],self._sim.data.get_body_xpos('cube')[0],self._sim.data.get_body_xpos('cube')[1],self._sim.data.get_body_xpos('cube')[2],self._sim.data.get_body_xquat('cube')[0],self._sim.data.get_body_xquat('cube')[1],self._sim.data.get_body_xquat('cube')[2],self._sim.data.get_body_xquat('cube')[3]])
                # print('positions:',self._positions)
            self._sim.step()
            self._viewer.render()


if __name__ == '__main__':
    experiment = "Single"
    for iteration in range(repeats):
        simulate = MUJOCO()
        simulate.set_current_iteration(iteration)
        simulate.set_experiment(experiment)
        simulate.run_mujoco()