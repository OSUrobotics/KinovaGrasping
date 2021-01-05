# This script is to teleoperate Kinova gripper in the mujoco env

import gym
import numpy as np
import pdb
from classifier_network import LinearNetwork, ReducedLinearNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import triang
#import serial
import matplotlib.pyplot as plt
#import optimizer
import csv
import time

def rand_vec():
    components = [np.random.normal() for i in range(3)]
    r = np.sqrt(sum(x*x for x in components))
    vec = [x/r for x in components]
    vec[2] = abs(vec[2])
    return vec

#Random stem oreintation
vec = rand_vec()
print("Random Vector: ", vec)

#Env setup
env = gym.make('gym_kinova_gripper:kinovagripper-v0')#,arm_or_end_effector="arm")
env.reset(shape_keys=["RGmBox"],obj_params=["RGCube","M"],hand_orientation="random")
#env.env._sim.data.set_joint_qvel('j2s7s300_joint_wrist',0)


action = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0])

t = 0
coords='local'

episode_obs=[]
value=0

s_or_f=[]
last_obs=[]
obs=[]
act=np.array([0.3,0.3,0.3,0])

#Actions [slide_x, slide_y, slide_z, finger_vel,finger_vel,finger_vel, wrist_vel]
actions=[[0,0,0,0.3,0.3,0.3, 0],[0,0,0,0.3,0.3,0,0],[0,0,0,0,0,0.3,0], [0,0,0,0,0,0,.3]]
poses=[[0.0,-0.03],[0.02,-0.03],[-0.02,-0.03],[-0.05,0],[-0.01,-0.035],[0.01,-0.035],[0.05,0],[-0.02,0.03],[0.0,0.03],[0.02,0.03]]

print("Action space: \n", env.action_space.sample())

#Grasp Classifier
#grasp_classifier = ReducedLinearNetwork()
#grasp_classifier=grasp_classifier.float()
#grasp_classifier.load_state_dict(torch.load('trained_model_05_30_20_1119local.pt'))
#grasp_classifier=grasp_classifier.float()
#grasp_classifier.eval()
#print('Grasp Classifier Loaded')

#What is this outer loop?
for f in range(3):

    #Episodes
    for k in range(10):
        #Setup env
        #Add random vector
        env.add_vec_site([0,0,0], vec)
        env.reset(shape_keys=["RGmBox"],obj_params=["RGCube","M"],hand_orientation="random")        
        print(env.env._sim.data.get_joint_qvel('j2s7s300_joint_wrist'))


        #Default action
        action=np.append([0,0,0],act)
        print('reset')

        #Timesteps
        for i in range(200):
            #Set action to lift
            if i == 150:
                print('move in z')
                action=np.array([0.15,0.05, 0.05, 0.05])
                env.env.pid=True
                last_obs.append(obs)

            #testing
            #if i ==50:
            #    action = np.array([0.0,0,0.15,0.3,0.3,0.3,0.3])
            
            #????
            if coords=='global':
                temp=np.array([action[0],action[1],action[2],1])
                action[0:3]=np.matmul(env.Twf[0:3,0:3],action[0:3])
            
            print("Action: ", action)
            #STEP
            obs, reward, done, _ = env.step(action)
            env.render()
            
            network_feed=obs[21:24]
            print('local obs',obs[21:24])
            network_feed=np.append(network_feed,obs[27:36])
            network_feed=np.append(network_feed,obs[49:51])
            states=torch.zeros(1,14, dtype=torch.float)
            for j in range(len(network_feed)):
                states[0][j]= network_feed[j]

            #Run grasp classifier

