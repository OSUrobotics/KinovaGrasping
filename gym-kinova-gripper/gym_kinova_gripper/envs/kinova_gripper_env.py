#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Kinova 3-fingered gripper in mujoco environment
# Summer 2019

###############

#TODO: Remove unecesssary commented lines
#TODO: Make a brief description of each function commented at the top of it

from gym import utils, spaces
import gym
import glfw
from gym.utils import seeding
# from gym.envs.mujoco import mujoco_env
import numpy as np
from mujoco_py import MjViewer, load_model_from_path, MjSim
import mujoco_py
# from PID_Kinova_MJ import *
import math
import matplotlib.pyplot as plt
import time
import os, sys
from scipy.spatial.transform import Rotation as R
import random
import pickle
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from classifier_network import LinearNetwork, ReducedLinearNetwork
import re
from scipy.stats import triang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KinovaGripper_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, arm_or_end_effector="hand", frame_skip=4):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        if arm_or_end_effector == "arm":
            self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300.xml")
            full_path = self.file_dir + "/kinova_description/j2s7s300.xml"
        elif arm_or_end_effector == "hand":
            pass
            #self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1.xml"
            self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_shg.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_shg.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mhg.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mhg.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bhg.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bhg.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_svase.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_svase.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mvase.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mvase.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bvase.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bvase.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcap.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bcap.xml"
            # full_path = file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml"
        else:
            print("CHOOSE EITHER HAND OR ARM")
            raise ValueError
        
        self._sim = MjSim(self._model)   # The simulator. This holds all the information about object locations and orientations
        self.Grasp_Reward=False   #This varriable says whether or not a grasp reward has  been given this run
        self._viewer = None   # The render window
        self.contacts=self._sim.data.ncon   # The number of contacts in the simulation environment
        self.Tfw=np.zeros([4,4])   # The trasfer matrix that gets us from the world frame to the local frame
        self.wrist_pose=np.zeros(3)  # The wrist position in world coordinates
        self.thetas=[0,0,0,0,0,0,0] # The angles of the joints of a real robot arm used for calculating the jacobian of the hand
        self._timestep = self._sim.model.opt.timestep 
        
        
        self._torque = [0,0,0,0] #Unused
        self._velocity = [0,0,0,0] #Unused
        self._jointAngle = [5,0,0,0] #Usused
        self._positions = [] # ??
        self._numSteps = 0
        self._simulator = "Mujoco"
        self.action_scale = 0.0333
        self.max_episode_steps = 150
        # Parameters for cost function
        self.state_des = 0.20 
        self.initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.action_space = spaces.Box(low=np.array([-0.8, -0.8, -0.8, -0.8]), high=np.array([0.8, 0.8, 0.8, 0.8]), dtype=np.float32) # Velocity action space
        self.const_T=np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])  #Transfer matrix from world frame to un-modified hand frame
        self.frame_skip = frame_skip # Used in step. Number of frames you go through before you reach the next step
        self.all_states = None  # This is the varriable we use to save the states before they are sent to the simulator when we are resetting.

        self.state_rep = "local" # change accordingly



        #In theory this is all unused --->
        min_hand_xyz = [-0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1, 0.0,-0.1, -0.1, 0.0, -0.1, -0.1, 0.0,-0.1, -0.1, 0.0, -0.1, -0.1, 0.0]
        min_obj_xyz = [-0.1, -0.01, 0.0]
        min_joint_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        min_obj_size = [0.0, 0.0, 0.0]
        min_finger_obj_dist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        min_obj_dot_prod = [0.0]
        min_f_dot_prod = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        max_hand_xyz = [0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5,0.1, 0.1, 0.5, 0.1, 0.1, 0.5,0.1, 0.1, 0.5, 0.1, 0.1, 0.5]
        max_obj_xyz = [0.1, 0.7, 0.5]
        max_joint_states = [0.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        max_obj_size = [0.5, 0.5, 0.5]
        max_finger_obj_dist = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]    
        max_obj_dot_prod = [1.0]
        max_f_dot_prod = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # print()
        if self.state_rep == "global" or self.state_rep == "local":

            obs_min = min_hand_xyz + min_obj_xyz + min_joint_states + min_obj_size + min_finger_obj_dist + min_obj_dot_prod #+ min_f_dot_prod
            obs_min = np.array(obs_min)
            # print(len(obs_min))

            obs_max = max_hand_xyz + max_obj_xyz + max_joint_states + max_obj_size + max_finger_obj_dist + max_obj_dot_prod #+ max_f_dot_prod 
            obs_max = np.array(obs_max)
            # print(len(obs_max))

            self.observation_space = spaces.Box(low=obs_min , high=obs_max, dtype=np.float32)
        elif self.state_rep == "metric":
            obs_min = list(np.zeros(17)) + [-0.1, -0.1, 0.0] + min_obj_xyz + min_joint_states + min_obj_size + min_finger_obj_dist + min_dot_prod
            obs_max = list(np.full(17, np.inf)) + [0.1, 0.1, 0.5] + max_obj_xyz + max_joint_states + max_obj_size + max_finger_obj_dist + max_dot_prod
            self.observation_space = spaces.Box(low=np.array(obs_min) , high=np.array(obs_max), dtype=np.float32)

        elif self.state_rep == "joint_states":
            obs_min = min_joint_states + min_obj_xyz + min_obj_size + min_dot_prod
            obs_max = max_joint_states + max_obj_xyz + max_obj_size + max_dot_prod
            self.observation_space = spaces.Box(low=np.array(obs_min) , high=np.array(obs_max), dtype=np.float32)
        # <---- end of unused section
        
        
        self.Grasp_net = LinearNetwork().to(device) # This loads the grasp classifier
        trained_model = "/home/orochi/KinovaGrasping/gym-kinova-gripper/trained_model_01_23_20_0111.pt"
        #trained_model = "/home/orochi/KinovaGrasping/gym-kinova-gripper/trained_model_01_23_20_2052local.pt"
        # self.Grasp_net = GraspValid_net(54).to(device) 
        # trained_model = "/home/graspinglab/NCS_data/ExpertTrainedNet_01_04_20_0250.pt"
        model = torch.load(trained_model)
        self.Grasp_net.load_state_dict(model)
        self.Grasp_net.eval()

    # Funtion to get 3D transformation matrix of the palm and get the wrist position and update both those varriables
    def _get_trans_mat_wrist_pose(self):  #WHY MUST YOU HATE ME WHEN I GIVE YOU NOTHING BUT LOVE?
        self.wrist_pose=np.copy(self._sim.data.get_geom_xpos('palm'))
<<<<<<< HEAD
        Rfa=self._sim.data.get_geom_xmat('palm')
        temp=np.matmul(Rfa,np.array([[0,0,1],[-1,0,0],[0,-1,0]]))
        temp=np.transpose(temp)
        temp=np.matmul(temp,np.array([[1,0,0],[0,1,0],[0,0,1]]))
        Tfa=np.zeros([4,4])
        Twf=np.zeros([4,4])
=======
        Rfa=np.copy(self._sim.data.get_geom_xmat('palm'))
        #print('xmat',Rfa)
        temp=np.matmul(Rfa,np.array([[0,0,1],[-1,0,0],[0,-1,0]]))
        temp=np.transpose(temp)
        #print('xmat times const matrix',temp)
#        temp=np.matmul(temp,np.array([[1,0,0],[0,1,0],[0,0,1]]))
        Tfa=np.zeros([4,4])
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
        Tfa[0:3,0:3]=temp
        Tfa[3,3]=1       
        Tfw=np.zeros([4,4])
        Tfw[0:3,0:3]=temp
        Tfw[3,3]=1
        self.wrist_pose=self.wrist_pose+np.matmul(np.transpose(Tfw[0:3,0:3]),[0.0,0.06,0.0])
        Tfw[0:3,3]=np.matmul(-np.transpose(Tfw[0:3,0:3]),self.wrist_pose)
        self.Tfw=Tfw 
<<<<<<< HEAD
        Twf[0:3,0:3]=np.transpose(Tfw[0:3,0:3])
        Twf[3,3]=1
        Twf[0:3,3]=-np.matmul(Twf[0:3,0:3],Tfw[0:3,3])
        self.Twf=np.linalg.inv(Twf)
=======
        self.Twf=np.linalg.inv(Tfw)
        #print('Tfw', self.Tfw)
        #print('Twf',self.Twf)
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
        
    def _get_jacobian(self): #(Currently Broken and Not Used)
        pi=np.pi
        R1=np.array([[1,0,0],[0,np.cos(self.thetas[0]),-np.sin(self.thetas[0])],[0,np.sin(self.thetas[0]),np.cos(self.thetas[0])]])
        R2=np.array([[np.cos(self.thetas[1]),0,np.sin(self.thetas[1])],[0,1,1],[-np.sin(self.thetas[1]),0,np.cos(self.thetas[1])]])
        R3=np.array([[1,0,0],[0,np.cos(self.thetas[2]),-np.sin(self.thetas[2])],[0,np.sin(self.thetas[2]),np.cos(self.thetas[2])]])
        R4=np.array([[np.cos(self.thetas[3]),0,np.sin(self.thetas[3])],[0,1,0],[-np.sin(self.thetas[3]),0,np.cos(self.thetas[3])]])
        R5=np.array([[1,0,0],[0,np.cos(self.thetas[4]), -np.sin(self.thetas[4])],[0,np.sin(self.thetas[4]),np.cos(self.thetas[4])]])
        R6=np.matmul(np.array([[np.cos(pi/3),-np.sin(pi/3),0],[np.sin(pi/3),np.cos(pi/3),0],[0,0,1]]),np.array([[1,0,0],[0,np.cos(self.thetas[5]),-np.sin(self.thetas[5])],[0,np.sin(self.thetas[5]),np.cos(self.thetas[5])]]))
        R7=np.matmul(np.array([[np.cos(-pi/3),-np.sin(-pi/3),0],[np.sin(-pi/3),np.cos(-pi/3),0],[0,0,1]]),np.array([[1,0,0],[0, np.cos(self.thetas[6]),-np.sin(self.thetas[6])],[0,np.sin(self.thetas[6]),np.cos(self.thetas[6])]]))
        #print("R7:, ", R7)
        L2=np.array([0.2755,0,0])
        L3=np.array([0.2755+0.210,0.0098,0])
        L4=np.array([.2755+.410,0.0098,0])
        L5=np.array([0.2755+0.410+.2073,0,0])
        L6=np.array([0.2755+0.410+.2073+0.0741*np.cos(pi/3),0.0741*np.sin(pi/3),0])
        L7=np.array([0.2755+0.410+.2073+0.0741*np.cos(pi/3)+0.0741*np.sin(pi/3),0.0741*np.cos(pi/3)+0.0741*np.sin(pi/3),0])
        
        S1=[1,0,0,0,0,0]
        S2=np.matmul(R1,[0,1,0])
        S2=np.append(S2,np.cross(-S2,L2))
        #print(S2)
        S3=np.matmul(np.matmul(R1,R2),[1,0,0])
        S3=np.append(S3,np.cross(-S3,L3))
        S4=np.matmul(np.matmul(R1,R2),np.matmul(R3,[0,1,0]))
        S4=np.append(S4,np.cross(-S4,L4))
        S5=np.matmul(np.matmul(R1,R2),np.matmul(np.matmul(R3,R4),[1,0,0]))
        S5=np.append(S5,np.cross(-S5,L5))
        S6=np.matmul(np.matmul(np.matmul(R1,R2),np.matmul(R3,R4)),np.matmul(R5,np.array([np.cos(pi/3),np.sin(pi/3),0])))
        S6=np.append(S6,np.cross(-S6,L6))
        S7=np.matmul(np.matmul(R1,R2),np.matmul(np.matmul(R3,R4),np.matmul(np.matmul(R5,R6),[1,0,0])))
        S7=np.append(S7,np.cross(-S7,L7))
        rotm=np.matmul(np.matmul(R1,R2),np.matmul(np.matmul(R3,R4),np.matmul(np.matmul(R5,R6),R7)))
        jaco=[S1,S2,S3,S4,S5,S6,S7]
        #print("jacobian", jaco)
        #print("rotation matrix",rotm)






    # Function to get the state of all the joints, including sliders
    def _get_joint_states(self):
        arr = []
        for i in range(len(self._sim.data.sensordata)-17):
            arr.append(self._sim.data.sensordata[i])

        return arr # it is a list

    # Function to return global or local transformation matrix
    def _get_obs(self, state_rep=None):  #TODO: Add or subtract elements of this to match the discussions with Ravi and Cindy
        if state_rep == None:
            state_rep = self.state_rep

        # states rep
        obj_pose = self._get_obj_pose()
        obj_pose = np.copy(obj_pose)
        self._get_trans_mat_wrist_pose()
        x_angle,z_angle = self._get_angles()
        joint_states = self._get_joint_states()
        obj_size = self._sim.model.geom_size[-1] 
        finger_obj_dist = self._get_finger_obj_dist()
        range_data=self._get_rangefinder_data()
        finger_joints = ["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
        
        fingers_6D_pose = []
        

        if state_rep == "global":#NOTE: only use local coordinates! global coordinates suck
            for joint in finger_joints:
                trans = self._sim.data.get_geom_xpos(joint)
                trans = list(trans)
                for i in range(3):
                    fingers_6D_pose.append(trans[i])
            fingers_6D_pose = fingers_6D_pose + list(self.wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [x_angle, z_angle] + range_data #+ [self.obj_shape]


        elif state_rep == "local":
            
            for joint in finger_joints:
<<<<<<< HEAD
                trans = self._sim.data.get_geom_xpos(joint)
=======
                trans = np.copy(self._sim.data.get_geom_xpos(joint))
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
                trans_for_roation=np.append(trans,1)
                trans_for_roation=np.matmul(self.Tfw,trans_for_roation)
                trans = trans_for_roation[0:3]
                trans = list(trans)
                for i in range(3):
                    fingers_6D_pose.append(trans[i])
            wrist_for_roation=np.append(self.wrist_pose,1)
            wrist_for_roation=np.matmul(self.Tfw,wrist_for_roation)
            wrist_pose = wrist_for_roation[0:3]
            obj_for_roation=np.append(obj_pose,1)
            obj_for_roation=np.matmul(self.Tfw,obj_for_roation)
            obj_pose = obj_for_roation[0:3]
            fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [x_angle, z_angle] + range_data #+ [self.obj_shape]
            #print('finger object distance',finger_obj_dist)
<<<<<<< HEAD
            print('distal joint states',joint_states[-3:])
=======
            #print('distal joint states',joint_states[-3:])
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
        elif state_rep == "joint_states":
            fingers_6D_pose = joint_states + list(obj_pose) + [obj_size[0], obj_size[1], obj_size[2]*2] + [x_angle, z_angle] #+ fingers_dot_prod
        return fingers_6D_pose 

    # Function to get the distance between the digits on the fingers and the object center
    def _get_finger_obj_dist(self): #TODO: check to see what happens when you comment out the dist[0]-= 0.0175 line and make sure it is outputting the right values
        finger_joints = ["f1_prox","f1_prox_1", "f2_prox", "f2_prox_1", "f3_prox", "f3_prox_1", "f1_dist", "f1_dist_1", "f2_dist", "f2_dist_1", "f3_dist", "f3_dist_1"]

        obj = self._get_obj_pose()
        dists = []
        for i in finger_joints:
            pos = self._sim.data.get_site_xpos(i)
            dist = np.absolute(pos[0:2] - obj[0:2])
            temp = np.linalg.norm(dist)
            dists.append(temp)
        return dists


    # get range data from 1 step of time 
    # Uncertainty: rangefinder could only detect distance to the nearest geom, therefore it could detect geom that is not object
    def _get_rangefinder_data(self):
        range_data = []
        for i in range(17):
            if self._sim.data.sensordata[i+len(self._sim.data.sensordata)-17]==-1:
                a=6
            else:
                a=self._sim.data.sensordata[i+len(self._sim.data.sensordata)-17]
            range_data.append(a)

        return range_data

    # Function to return the object position in world coordinates
    def _get_obj_pose(self):
        arr = self._sim.data.get_geom_xpos("object")
        return arr
    
    # Function to return the angles between the palm normal and the object location
    def _get_angles(self):
<<<<<<< HEAD
=======
        #t=time.time()
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
        obj_pose = self._get_obj_pose()
        self._get_trans_mat_wrist_pose()
        local_obj_pos=np.copy(obj_pose)
        local_obj_pos=np.append(local_obj_pos,1)
        local_obj_pos=np.matmul(self.Tfw,local_obj_pos)
        obj_wrist = local_obj_pos[0:3]/np.linalg.norm(local_obj_pos[0:3])
        center_line = np.array([0,1,0])
        z_dot = np.dot(obj_wrist[0:2],center_line[0:2])
        z_angle = np.arccos(z_dot/np.linalg.norm(obj_wrist[0:2]))
        x_dot = np.dot(obj_wrist[1:3],center_line[1:3])
        x_angle = np.arccos(x_dot/np.linalg.norm(obj_wrist[1:3]))
<<<<<<< HEAD
=======
        #print('angle calc took', t-time.time(), 'seconds')
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
        return x_angle,z_angle

    # Function to get rewards based only on the lift reward. This is primarily used to generate data for the grasp classifier
    def _get_reward_DataCollection(self):
        obj_target = 0.2
        obs = self._get_obs(state_rep="global") 
        # TODO: change obs[23] and obs[5] to the simulator height object
        if abs(obs[23] - obj_target) < 0.005 or (obs[23] >= obj_target):  #Check to make sure that obs[23] is still the object height. Also local coordinates are a thing
            lift_reward = 1
            done = True
        elif obs[5]>obj_target+0.05:
            lift_reward=0.0
            done=True
        else:
            lift_reward = 0
            done = False        
        return lift_reward, {}, done

    # Function to get rewards for RL training
    def _get_reward(self): # TODO: change obs[23] and obs[5] to the simulator height object and stop using _get_obs
        #TODO: Make sure this works with the new grasp classifier

        # object height target
        obj_target = 0.2

        # Grasp reward
        grasp_reward = 0.0
        obs = self._get_obs(state_rep="global") 
        loc_obs=self._get_obs()
        
        network_inputs=obs[0:5]
        network_inputs=np.append(network_inputs,obs[6:23])
        network_inputs=np.append(network_inputs,obs[24:])
        inputs = torch.FloatTensor(np.array(network_inputs)).to(device)
        
        if np.max(np.array(obs[41:47])) < 0.035 or np.max(np.array(obs[35:41])) < 0.015: 
             outputs = self.Grasp_net(inputs).cpu().data.numpy().flatten()
             if (outputs >=0.3) & (not self.Grasp_Reward):
                 grasp_reward = 5.0
                 self.Grasp_Reward=True
             else:
                 grasp_reward = 0.0

        if abs(obs[23] - obj_target) < 0.005 or (obs[23] >= obj_target):
            lift_reward = 50.0
            done = True
        else:
            lift_reward = 0.0
            done = False

        finger_reward = -np.sum((np.array(obs[41:47])) + (np.array(obs[35:41])))

        reward = 0.2*finger_reward + lift_reward + grasp_reward

        return reward, {}, done

    # only set proximal joints, cuz this is an underactuated hand
    def _set_state(self, states):
        self._sim.data.qpos[0] = states[0]
        self._sim.data.qpos[1] = states[1]
        self._sim.data.qpos[3] = states[2]
        self._sim.data.qpos[5] = states[3]
        self._sim.data.set_joint_qpos("object", [states[4], states[5], states[6], 1.0, 0.0, 0.0, 0.0])
        self._sim.forward()
<<<<<<< HEAD
=======
        
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
    # Function to get the dimensions of the object
    def _get_obj_size(self): #TODO: tweak this so that it doesn't get fucked by a shape with only two dimensions (eg. cylinder)
        size=self._sim.model.geom_size[-1]
        if size[2]==0:
            size[2]=size[1]
        return size

    # Function to place the object at a random position near the hand with a probability density designed to create more difficult grasps
    def randomize_initial_pose(self, collect_data, size, shape):  #This will get fixed by Stephanie
        geom_size = size

        if geom_size == "s":
            if not collect_data:
                x = [0.05, 0.04, 0.03, 0.02, -0.05, -0.04, -0.03, -0.02]
                y = [0.0, 0.02, 0.03, 0.04]
                rand_x = random.choice(x)
                rand_y = 0.0                
                if rand_x == 0.05 or rand_x == -0.05:
                    rand_y = 0.0
                elif rand_x == 0.04 or rand_x == -0.04:
                    rand_y = random.uniform(0.0, 0.02)
                elif rand_x == 0.03 or rand_x == -0.03:
                    rand_y = random.uniform(0.0, 0.03)
                elif rand_x == 0.02 or rand_x == -0.02:
                    rand_y = random.uniform(0.0, 0.04)
            else:
                x = [0.04, 0.03, 0.02, -0.04, -0.03, -0.02]
                y = [0.0, 0.02, 0.03, 0.04]
                rand_x = random.choice(x)
                rand_y = 0.0                    
                if rand_x == 0.04 or rand_x == -0.04:
                    rand_y = random.uniform(0.0, 0.02)
                elif rand_x == 0.03 or rand_x == -0.03:
                    rand_y = random.uniform(0.0, 0.03)
                elif rand_x == 0.02 or rand_x == -0.02:
                    rand_y = random.uniform(0.0, 0.04)                
        if geom_size == "m":
            if shape =='b':
                x=[0.04,0.05,-0.04,-0.05]
                y=0.0
            else:
                x=[0.03,0.04,-0.03,-0.04]
                y=0.0
            rand_x = random.choice(x)
            rand_y = 0.0
            if rand_x == 0.04 or rand_x == -0.04:
                rand_y = 0.0
            elif rand_x == 0.03 or rand_x == -0.03:
                rand_y = random.uniform(0.0, 0.02)
            elif rand_x == 0.02 or rand_x == -0.02:
                rand_y = random.uniform(0.0, 0.03)
        if geom_size == "b":
            x = [0.03, 0.02, -0.03, -0.02]
            y = [0.0, 0.02]
            rand_x = random.choice(x)
            rand_y = 0.0
            if rand_x == 0.03 or rand_x == -0.03:
                rand_y = 0.0
            elif rand_x == 0.02 or rand_x == -0.02:
                rand_y = random.uniform(0.0, 0.02)
        return rand_x, rand_y
        
    # Function to run all the experiments for RL training
    def experiment(self, exp_num, stage_num):
        objects = {}

        # ------ Experiment 1 ------- #
        if exp_num == 1:
                
            # Exp 1 Stage 1: Change size ---> 
            if stage_num == 1:
                objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
                objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"

                # Testing Exp 1 Stage 1
                #objects["mbox"] = "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"

            # Exp 1 Stage 2: Change shape
            if stage_num == 2:
                objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
                objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"
                objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
                objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
                # Testing Exp 1 Stage 2
                #objects["mcyl"] = "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
                #objects["mbox"] = "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
        # ------ Experiment 2 ------- #
        elif exp_num == 2:
            # Exp 2 Stage 1: Change shape
            if stage_num == 1:
                objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
                objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"

            # Exp 2 Stage 2: Change size
            if stage_num == 2:
                objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
                objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"
                objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
                objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
                
            # Testing Exp 2
            # objects["mbox"] = "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
            # objects["mcyl"] = "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"

        # ------ Experiment 3 ------ #
        elif exp_num == 3:
            # Mix all
            objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
            objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
            objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
            objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"
            
            # Testing Exp 3
            # objects["mbox"] = "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
            # objects["mcyl"] = "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"

        else:
            print("Enter Valid Experiment Number")
            raise ValueError

        return objects


    def randomize_all(self): #Stephanie has a new version, will merge
        # objects = {}
        # objects["sbox"] = "/kinova_description/j2s7s300_end_effector.xml"
        # objects["mbox"] = "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
        # objects["bbox"] = "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
        # objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
        # objects["mcyl"] = "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
        # objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"

        objects = self.experiment(2, 2)

        random_shape = np.random.choice(list(objects.keys()))
        #self._model = load_model_from_path(self.file_dir + objects[random_shape])
        if (random_shape=="sbox") |(random_shape=="mbox")|(random_shape=="bbox"):
            self.obj_shape=1
        elif (random_shape=="scyl")|(random_shape=="mcyl")|(random_shape=="bcyl"):
            self.obj_shape=0
        self._sim = MjSim(self._model)
        # print (random_shape)
        if random_shape == "sbox" or random_shape == "scyl":
            x, y = self.randomize_initial_pose(False, "s",random_shape[1])
            z = 0.05
        elif random_shape == "mbox" or random_shape == "mcyl":
            x, y = self.randomize_initial_pose(False, "m",random_shape[1])
            z = 0.055
        elif random_shape == "bbox" or random_shape == "bcyl":
            x, y = self.randomize_initial_pose(False, "b",random_shape[1])
            z = 0.06            
        else:
            print("size and shape are incorrect")
            raise ValueError

        return x, y, z

    #Function to randomize the position of the object for grasp classifier data collection
    def randomize_initial_pos_data_collection(self):
        size=self._get_obj_size()
        #The old way to generate random poses
        rand_x=triang.rvs(0.5)
        rand_x=(rand_x-0.5)*(0.16-2*size[0])
        rand_y=np.random.uniform()
        if rand_x>=0:
            rand_y=rand_y*(-(0.07-size[0]*np.sqrt(2))/(0.08-size[0])*rand_x+(0.07-size[0]))
        else:
            rand_y=rand_y*((0.07-size[0]*np.sqrt(2))/(0.08-size[0])*rand_x+(0.07-size[0]))
        z = self._sim.model.geom_size[-1][-1]
        if z == 0:
            z=self._sim.model.geom_size[-1][-2]
        return rand_x, rand_y, z    

    #Function to reset the simulator
    def reset(self,start_pos=None,obj_params=None):
        # x, y = self.randomize_initial_pose(False, "s") # for RL training
        #x, y = self.randomize_initial_pose(True) # for data collection
        objects = self.experiment(1, 2)
        
        shapes=list(objects.keys())
        #print(shapes[0])
        #self._get_jacobian()
        '''
        hand_rotation=np.random.normal(0,0.125,3)
        '''
        
        #hand_rotation=np.array([np.pi/2, 0, np.pi/2])
        
        #Might make sense to move this to experiments and use that to generate the objects and positions
        if obj_params !=None:
            if obj_params[0] == "Box":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
                    #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_svase.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_svase.xml"
                    #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mvase.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mvase.xml"
                    #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bvase.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bvase.xml"
                    #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcap.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bcap.xml"
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
                elif obj_params[1] == "S":
                    self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1.xml"
            elif obj_params[0] == "Cylinder":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"          
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
                elif obj_params[1] == "S":
                     self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
            elif obj_params[0] == "Hour":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bhg.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bhg.xml"
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mhg.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mhg.xml"
                elif obj_params[1] == "S":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_shg.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_shg.xml"
        
        self._model = load_model_from_path(self.file_dir + self.filename)

        #TODO: This needs a separate testing function        
        '''
        if start_pos==None:
            x, y, z = self.randomize_all() # for RL training all objects
        elif start_pos<23:
            x=(start_pos)*0.005-0.055
            y=0.0
            z=self._sim.model.geom_size[-1][-1]
            #self.obj_shape=1
            #self._model = load_model_from_path(self.file_dir + objects[shapes[1]])
            print(start_pos,'  ',x)
        elif start_pos>=23:
            x=(start_pos-23)*0.005-0.04
            y=0.0
            z=self._sim.model.geom_size[-1][-1]
            #self.obj_shape=0
            #self._model = load_model_from_path(self.file_dir + objects[shapes[0]])
            print(start_pos,'  ',x)
        '''
        self._sim = MjSim(self._model)    
        if start_pos is None:
            x, y, z = self.randomize_initial_pos_data_collection()
        else:
            x, y, z = start_pos[0], start_pos[1], start_pos[2]
            
        self.all_states_1 = np.array([0.0, 0.0, 0.0, 0.0, x, y, z])
        self.Grasp_Reward=False
        self.all_states_2 = np.array([0.0, 0.9, 0.9, 0.9, 0.0, -0.01, 0.05])
        self.all_states = [self.all_states_1 , self.all_states_2] 

        self._set_state(self.all_states[0])


        states = self._get_obs()
        obj_pose=self._get_obj_pose()
        deltas=[x-obj_pose[0],y-obj_pose[1],z-obj_pose[2]]
        
        
        if np.linalg.norm(deltas)>0.05:
            self.all_states_1=np.array([0.0, 0.0, 0.0, 0.0, x+deltas[0], y+deltas[1], z+deltas[2]])
            self.all_states=[self.all_states_1,self.all_states_2]
            self._set_state(self.all_states[0])
            states = self._get_obs()
        
        #These two varriables are used when the action space is in joint states
        self.t_vel = 0
        self.prev_obs = []
        
        return states

    #Function to display the current state in a video. The video is always paused when it first starts up.
    def render(self, mode='human'): #TODO: Fix the rendering issue where a new window gets built every time the environment is reset or the window freezes when it is reset
        a=False
        if self._viewer is None:
            self._viewer = MjViewer(self._sim)
            a=True
        self._viewer.render()
        if a:
            self._viewer._paused=True
            
    #Function to close the rendering window
    def close(self): #This doesn't work right now
        if self._viewer is not None:
            self._viewer = None
    
    #Function to pause the rendering video
    def pause(self):
        self._viewer._paused=True
    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ###################################################
    ##### ---- Action space : Joint Velocity ---- #####
    ###################################################
    #Function to step the simulator forward in time
    def step(self, action): #TODO: fix this so that it doesn't jerk around when sliding and so that we can rotate the hand
        total_reward = 0
        
        #if len(action)==4:
        #    action=[action[0],0,0,0,0,0,action[1],action[2],action[3]]
        self._get_trans_mat_wrist_pose()
        
        # This is used to tell if the object is in contact with the hand and which parts of the hand
        '''
        tot_external_force=np.array([0,0,0])
        #flag = True
        for i in range(self._sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = self._sim.data.contact[i]
            temp1=not((contact.geom1==0) | (contact.geom2==0))
            temp2=((contact.geom1==8)|(contact.geom2==8))
            if temp1 & temp2: 
                #flag=False
                #print('contact:', i)
                #print('distance:', contact.dist)
                #print('geom1:', contact.geom1, self._sim.model.geom_id2name(contact.geom1))
                #print('geom2:', contact.geom2, self._sim.model.geom_id2name(contact.geom2))
                #print(not((contact.geom1==0) | (contact.geom2==0)))
                #print(((contact.geom1==8)|(contact.geom2==8)))
                #print(not((contact.geom1==0) | (contact.geom2==0))&((contact.geom1==8)|(contact.geom2==8)))
                #print('contact position:', contact.pos)
                
                # Use internal functions to read out mj_contactForce
                c_array = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(self._sim.model, self._sim.data, i, c_array)
                
                
                # Convert the contact force from contact frame to world frame
                ref = np.reshape(contact.frame, (3,3))
                c_force = np.dot(np.linalg.inv(ref), c_array[0:3])
                #c_torque = np.dot(np.linalg.inv(ref), c_array[3:6])
                #print('contact force in world frame:', c_force)  
                #print('contact torque in world frame:', c_torque)
                #print()
                
                tot_external_force=tot_external_force+c_force
        #print(tot_external_force)
        '''
        
        
        for _ in range(self.frame_skip): #TODO: Fix this for local coordinates
<<<<<<< HEAD
            slide_vector=np.array([action[0],action[1],action[2]])
=======
            slide_vector=np.array([-action[0],-action[1],action[2]])
            #print(slide_vector)
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
            #print("slide vector", slide_vector)
            #print('pre',slide_vector)
            #slide_2=np.matmul(self.Twf[0:3,0:3],slide_vector)
            #slide_vector=np.matmul(self.Twf[0:3,0:3],slide_vector)
<<<<<<< HEAD
=======
            #print('slide_vector',slide_vector)
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
            #print('TFW',self.Tfw)
            #print('TWF', self.Twf)
            #print('TFW slide',slide_vector)
            #print('TWF slide', slide_2)
<<<<<<< HEAD
            #print('post',slide_vector)
=======
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
            '''
            if action[0] < 0.0:
                self._sim.data.ctrl[0] = 0.0
            else:    
                #self._sim.data.ctrl[0] = (action[0] / 0.8) * 0.2
                self._sim.data.ctrl[0] = action[0]
            '''
            
            for i in range(3):
                # vel = action[i]
                
                #if action[i+1] < 0.0:
                #self._sim.data.ctrl[i+1] = 0.0
                    
                
                #else:    
                #self._sim.data.ctrl[(i)*2+1] = -tot_external_force[i]/24
                self._sim.data.ctrl[(i)*2] = slide_vector[i]
                self._sim.data.ctrl[i+6] = action[i+3]
            #self._sim.data.ctrl[-2] = 3000
            #self._sim.data.ctrl[-1]=5*hand_pos[-1]
            self._sim.step()
        obs = self._get_obs()

        ### Get this reward for RL training ###
        #total_reward, info, done = self._get_reward()

        ### Get this reward for grasp classifier collection ###
        total_reward, info, done = self._get_reward_DataCollection()
        return obs, total_reward, done, info
    
    
    #TODO: Make a config file that makes it easy to switch action spaces and set global varriables correctly
    
    #####################################################

    ###################################################
    ##### ---- Action space : Joint Angle ---- ########
    ###################################################
    # def step(self, action):
    #     total_reward = 0
    #     for _ in range(self.frame_skip):
    #         self.pos_control(action)
    #         self._sim.step()

    #     obs = self._get_obs()
    #     total_reward, info, done = self._get_reward()
    #     self.t_vel += 1
    #     self.prev_obs.append(obs)
    #     # print(self._sim.data.qpos[0], self._sim.data.qpos[1], self._sim.data.qpos[3], self._sim.data.qpos[5])
    #     return obs, total_reward, done, info

    # def pos_control(self, action):
    #     # position 
    #     # print(action)

    #     self._sim.data.ctrl[0] = (action[0] / 1.5) * 0.2
    #     self._sim.data.ctrl[1] = action[1]
    #     self._sim.data.ctrl[2] = action[2]
    #     self._sim.data.ctrl[3] = action[3]
    #     # velocity 
    #     if abs(action[0] - 0.0) < 0.0001:
    #         self._sim.data.ctrl[4] = 0.0
    #     else:
    #         self._sim.data.ctrl[4] = 0.1
    #         # self._sim.data.ctrl[4] = (action[0] - self.prev_action[0] / 25)        

    #     if abs(action[1] - 0.0) < 0.001:
    #         self._sim.data.ctrl[5] = 0.0
    #     else:
    #         self._sim.data.ctrl[5] = 0.01069
    #         # self._sim.data.ctrl[5] = (action[1] - self.prev_action[1] / 25)    

    #     if abs(action[2] - 0.0) < 0.001:
    #         self._sim.data.ctrl[6] = 0.0
    #     else:
    #         self._sim.data.ctrl[6] = 0.01069
    #         # self._sim.data.ctrl[6] = (action[2] - self.prev_action[2] / 25)    

    #     if abs(action[3] - 0.0) < 0.001:
    #         self._sim.data.ctrl[7] = 0.0                        
    #     else:
    #         self._sim.data.ctrl[7] = 0.01069
    #         # self._sim.data.ctrl[7] = (action[3] - self.prev_action[3] / 25)    
    
        # self.prev_action = np.array([self._sim.data.qpos[0], self._sim.data.qpos[1], self._sim.data.qpos[3], self._sim.data.qpos[5]])
        # self.prev_action = np.array([self._sim.data.qpos[0], self._sim.data.qpos[1], self._sim.data.qpos[3], self._sim.data.qpos[5]])

    #####################################################


class GraspValid_net(nn.Module):
    def __init__(self, state_dim):
        super(GraspValid_net, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        # pdb.set_trace()

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a =    torch.sigmoid(self.l3(a))
        return a
