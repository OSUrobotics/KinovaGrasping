# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:55:07 2018

@author: jack
"""
from PID import *
import subprocess
import time

def simSteps(experiment,timestep):      
    if experiment == "Single":
        seconds = 6
    elif experiment == "Double":
        seconds = 20
    elif experiment == "Cube":
        seconds = 20
        
    return int(seconds/timestep)
        
def set_target_thetas(num_steps, pid, experiment,simulator, simStep):
    if experiment == "Single":
        if simStep == 0:
            pid[1].set_target_theta(-100)
        else:
            return pid
            
    elif experiment == "Double":
        if simStep == 0:
            pid[1].set_target_theta(-90)
            pid[4].set_target_theta(-90)
        elif simStep % (num_steps*0.5) == 0:
            pid[1].set_target_theta(-90)
            pid[4].set_target_theta(90)
        elif simStep % (num_steps*0.25) == 0:
            pid[1].set_target_theta(0)
            pid[4].set_target_theta(0)
        else:
            return pid

    elif experiment == "Cube":
        if simStep == 0:
            pid[1].set_target_theta(30)
            pid[2].set_target_theta(120)
            pid[3].set_target_theta(90)
        elif  num_steps*0.4 == simStep:
            pid[1].set_target_theta(-60)
            pid[2].set_target_theta(65)
            pid[3].set_target_theta(90)
        elif num_steps*0.7 == simStep:
            pid[1].set_target_theta(-67)
            pid[2].set_target_theta(30)
            pid[3].set_target_theta(90)
        else:
            return pid
    if simulator == "PyBullet":
        pid = pid[-2:] + pid[0:-2]

    return pid
    
def open_vrep(vrep_path):
    terminal_string = "cd " + vrep_path + " && ./vrep.sh"
    proc = subprocess.Popen([terminal_string], shell=True)
    time.sleep(8)
    proc.terminate()

def saveStats(experiment, iteration, physics_engine, simulator, positions):
    #Save simulation Data     
    fileString = 'Results/%s_%s%s_%d.csv'%(experiment,simulator,physics_engine,iteration)
    with open(fileString, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        for xyz in positions:
            writer.writerow(xyz)