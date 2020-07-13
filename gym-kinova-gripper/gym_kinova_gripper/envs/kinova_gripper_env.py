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
        self.arm_or_hand=arm_or_end_effector
        if arm_or_end_effector == "arm":
            self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300.xml")
            full_path = self.file_dir + "/kinova_description/j2s7s300.xml"
            self.filename= "/kinova_description/j2s7s300.xml"
        elif arm_or_end_effector == "hand":
            pass
            #self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1.xml"
            #self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_shg.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_shg.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mhg.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mhg.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bhg.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bhg.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_svase.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_svase.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mvase.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mvase.xml"
            self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bvase.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bvase.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcap.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bcap.xml"
            #full_path = file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_blemon.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_blemon.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bRectBowl.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bRectBowl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bRoundBowl.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bRoundBowl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bbottle.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bbottle.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_btbottle.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_btbottle.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_slemon.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_slemon.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_sRectBowl.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_sRectBowl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_sRoundBowl.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_sRoundBowl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_sbottle.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_sbottle.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_stbottle.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_stbottle.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mlemon.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mlemon.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mRectBowl.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mRectBowl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mRoundBowl.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mRoundBowl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mbottle.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mbottle.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mtbottle.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mtbottle.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_msphere.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_sphere.xml"

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
        
        self.step_coords='global'
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
        trained_model = "/home/orochi/KinovaGrasping/gym-kinova-gripper/trained_model_05_28_20_2105local.pt"
        #trained_model = "/home/orochi/KinovaGrasping/gym-kinova-gripper/trained_model_01_23_20_2052local.pt"
        # self.Grasp_net = GraspValid_net(54).to(device) 
        # trained_model = "/home/graspinglab/NCS_data/ExpertTrainedNet_01_04_20_0250.pt"
        model = torch.load(trained_model)
        self.Grasp_net.load_state_dict(model)
        self.Grasp_net.eval()
        obj_list=['Coords_try1.txt','Coords_CubeM.txt','Coords_try1.txt','Coords_CubeB.txt','Coords_CubeM.txt','Coords_CubeS.txt']
        self.random_poses=[[],[],[],[],[],[]]
        for i in range(len(obj_list)):
            random_poses_file=open(obj_list[i],"r")
            #temp=random_poses_file.read()
            lines_list = random_poses_file.readlines()
            temp = [[float(val) for val in line.split()] for line in lines_list[1:]]
            self.random_poses[i]=temp
            random_poses_file.close()
        self.instance=0#int(np.random.uniform(low=0,high=100))
    
    # Funtion to get 3D transformation matrix of the palm and get the wrist position and update both those varriables
    def _get_trans_mat_wrist_pose(self):  #WHY MUST YOU HATE ME WHEN I GIVE YOU NOTHING BUT LOVE?
        self.wrist_pose=np.copy(self._sim.data.get_geom_xpos('palm')) 
        Rfa=np.copy(self._sim.data.get_geom_xmat('palm'))
        #print('wrist pose in xpos',self.wrist_pose)
        temp=np.matmul(Rfa,np.array([[0,0,1],[-1,0,0],[0,-1,0]]))
        temp=np.transpose(temp)
        #print('xmat times const matrix',temp)
        #temp=np.matmul(temp,np.array([[1,0,0],[0,1,0],[0,0,1]]))
        Tfa=np.zeros([4,4])
        Tfa[0:3,0:3]=temp
        Tfa[3,3]=1       
        Tfw=np.zeros([4,4])
        Tfw[0:3,0:3]=temp
        Tfw[3,3]=1
        self.wrist_pose=self.wrist_pose+np.matmul(np.transpose(Tfw[0:3,0:3]),[0.0,0.06,0.0])
        Tfw[0:3,3]=np.matmul(-(Tfw[0:3,0:3]),self.wrist_pose)
        self.Tfw=Tfw 
        self.Twf=np.linalg.inv(Tfw)
        #print('Tfw', np.round(self.Tfw,decimals=2))
        #print('Twf',np.round(self.Twf,decimals=2))
    
    def experimental_sensor(self,rangedata,finger_pose,gravity):
        #print('flimflam')
        #finger_joints = ["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
        finger_pose=np.array(finger_pose)
        
        s1=finger_pose[0:3]-finger_pose[6:9]
        s2=finger_pose[0:3]-finger_pose[3:6]
        front_area=np.linalg.norm(np.cross(s1,s2))/2
        top1=np.linalg.norm(np.cross(finger_pose[0:3],finger_pose[9:12]))/2
        top2=np.linalg.norm(np.cross(finger_pose[9:12],finger_pose[12:15]))/2
        top3=np.linalg.norm(np.cross(finger_pose[3:6],finger_pose[12:15]))/2
        top4=np.linalg.norm(np.cross(finger_pose[6:9],finger_pose[15:18]))/2
        top5=np.linalg.norm(np.cross(finger_pose[9:12],finger_pose[15:18]))/2
        total1=top1+top2+top3
        total2=top1+top4+top5
        top_area=max(total1,total2)
        #print('front',front_area,'top',top_area)
        
        sites=["palm","palm_1","palm_2","palm_3","palm_4"]
        obj_pose=[]#np.zeros([5,3])
        xs=[]
        ys=[]
        zs=[]
        for i in range(len(sites)):
            temp=self._sim.data.get_site_xpos(sites[i])
            temp=np.append(temp,1)
            temp=np.matmul(self.Tfw,temp)
            temp=temp[0:3]
            if rangedata[i] < 0.06:
                temp[1]+=rangedata[i]
                obj_pose=np.append(obj_pose,temp)
            #obj_pose[i,:]=temp
        for i in range(int(len(obj_pose)/3)):
            xs=np.append(xs,obj_pose[i*3])
            ys=np.append(ys,obj_pose[i*3+1])
            zs=np.append(zs,obj_pose[i*3+2])
        if xs ==[]:
            sensor_pose=[0.2,0.2,0.2]
        else:
            sensor_pose=[np.average(xs),np.average(ys),np.average(zs)]
        #print('object pose',obj_pose)
        #print('finger pose',finger_pose)
        #print('sensed pos',sensor_pose)
        obj_size=np.copy(self._get_obj_size())
        if np.argmax(np.abs(gravity))==2:
            front_part=np.abs(obj_size[0]*obj_size[2])/front_area
            top_part=np.abs(obj_size[0]*obj_size[1])/top_area
        elif np.argmax(np.abs(gravity))==1:
            front_part=np.abs(obj_size[0]*obj_size[2])/front_area
            top_part=np.abs(obj_size[1]*obj_size[2])/top_area
        else:
            front_part=np.abs(obj_size[0]*obj_size[1])/front_area
            top_part=np.abs(obj_size[0]*obj_size[2])/top_area 

        #print(front_part*front_area,front_area)
        #print(top_part*top_area,top_area)
        #print('sensor pos', sensor_pose)
        #print('front and top part', front_part, top_part)
        return sensor_pose,front_part, top_part
        
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




    def get_sim_state(self): #this gives you the whole damn qpos
        return np.copy(self._sim.data.qpos)
    
    def set_sim_state(self,qpos,obj_state):#this just sets all the qpos of the simulation manually. Is it bad? Probably. Do I care at this point? Not really
        self._sim.data.set_joint_qpos("object", [obj_state[0], obj_state[1], obj_state[2], 1.0, 0.0, 0.0, 0.0])
        for i in range(len(self._sim.data.qpos)):
            self._sim.data.qpos[i]=qpos[i]
        self._sim.forward()
        
    # Function to get the state of all the joints, including sliders
    def _get_joint_states(self):
        arr = []
        for i in range(len(self._sim.data.sensordata)-17):
            arr.append(self._sim.data.sensordata[i])
        arr[0]=-arr[0]
        arr[1]=-arr[1]
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
        obj_size = self._get_obj_size()
        finger_obj_dist = self._get_finger_obj_dist()
        range_data=self._get_rangefinder_data()
        finger_joints = ["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
        gravity=[0,0,-1]
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
                trans = np.copy(self._sim.data.get_geom_xpos(joint))
                trans_for_roation=np.append(trans,1)
                trans_for_roation=np.matmul(self.Tfw,trans_for_roation)
                trans = trans_for_roation[0:3]
                trans = list(trans)
                for i in range(3):
                    fingers_6D_pose.append(trans[i])
            #print('world wrist pose',self.wrist_pose)
            wrist_for_rotation=np.append(self.wrist_pose,1)
            wrist_for_rotation=np.matmul(self.Tfw,wrist_for_rotation)
            
            wrist_pose = wrist_for_rotation[0:3]
            obj_for_roation=np.append(obj_pose,1)
            obj_for_roation=np.matmul(self.Tfw,obj_for_roation)
            obj_pose = obj_for_roation[0:3]
            gravity=np.matmul(self.Tfw[0:3,0:3],gravity)
            sensor_pos,front_thing,top_thing=self.experimental_sensor(range_data,fingers_6D_pose,gravity)
            #print('center of object',obj_pose)
            
            #print('the stuff',sensor_pos, front_thing, top_thing)
            #print('pose', fingers_6D_pose)
            #print('wrist', wrist_pose)
            #print('obj', obj_pose)
            #print('joint states', joint_states)
            #print('object size', obj_size)
            #print('finger obj dist', finger_obj_dist)
            #print('x and z', [x_angle, z_angle])
            #print('range data', range_data)
            #print('gravituy',gravity)
            #print('sensor pos', sensor_pos)
            #print('ratios', front_thing, top_thing)
            fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [x_angle, z_angle] + range_data + [gravity[0],gravity[1],gravity[2]] + [sensor_pos[0],sensor_pos[1],sensor_pos[2]] + [front_thing, top_thing] #+ [self.obj_shape]
            #print('everything', fingers_6D_pose)
            #print('distal joint states',joint_states[-3:])
            #print(np.shape(fingers_6D_pose))
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
        #t=time.time()
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
        #print('angle calc took', t-time.time(), 'seconds')
        return x_angle,z_angle

    # Function to get rewards based only on the lift reward. This is primarily used to generate data for the grasp classifier
    def _get_reward_DataCollection(self):
        obj_target = 0.2
        obs = self._get_obs(state_rep="global")
        # TODO: change obs[23] and obs[5] to the simulator height object
        if abs(obs[23] - obj_target) < 0.005 or (obs[23] >= obj_target):  #Check to make sure that obs[23] is still the object height. Also local coordinates are a thing
            lift_reward = 1
            done = True
        elif obs[20]>obj_target+0.2:
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
    #we have a problem here (a binomial in the denomiator)
    #ill use the quotient rule
    def _set_state(self, states):
        #print('sensor data',self._sim.data.sensordata[0:9])
        #print('qpos',self._sim.data.qpos[0:9])
        #print('states',states)
        self._sim.data.qpos[0] = states[0]
        self._sim.data.qpos[1] = states[1]
        self._sim.data.qpos[2] = states[2]
        self._sim.data.qpos[3] = states[3]
        self._sim.data.qpos[5] = states[4]
        self._sim.data.qpos[7] = states[5]
        self._sim.data.set_joint_qpos("object", [states[6], states[7], states[8], 1.0, 0.0, 0.0, 0.0])
        self._sim.forward()
        
    # Function to get the dimensions of the object
    def _get_obj_size(self):
        #TODO: fix this shit
        num_of_geoms=np.shape(self._sim.model.geom_size)
        final_size=[0,0,0]
        for i in range(num_of_geoms[0]-8):
            size=np.copy(self._sim.model.geom_size[-1-i])
            if size[2]==0:
                size[2]=size[1]
                size[1]=size[0]
            if ('lemon' in self.filename)|('Bowl' in self.filename):
                temp=size[0]
                size[0]=size[2]
                size[2]=temp
            if 'Bowl' in self.filename:
                if 'Rect' in self.filename:
                    final_size[0]=0.17
                    final_size[1]=0.17
                    final_size[2]=0.07
                else:
                    final_size[0]=0.175
                    final_size[1]=0.175
                    final_size[2]=0.065
                if self.obj_size=='m':
                    for j in range(3):
                        final_size[j]=final_size[j]*0.85
                elif self.obj_size=='s':
                    for j in range(3):
                        final_size[j]=final_size[j]*0.7
            else:
                final_size[0]=max(size[0],final_size[0])
                final_size[1]=max(size[1],final_size[1])
                final_size[2]+=size[2]
        return final_size

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
    def experiment(self, exp_num, stage_num): #TODO: Talk to people thursday about adding the hourglass and bottles to this dataset.
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
    def randomize_initial_pos_data_collection(self,orientation="side",obj=0):
        
        size=self._get_obj_size()
        #The old way to generate random poses
        if orientation=='side':
            '''
            temp=self.random_poses[obj][self.instance]
            rand_x=temp[0]
            rand_y=temp[1]
            z=temp[2]
            self.instance+=1
            '''
            rand_x=triang.rvs(0.5)
            rand_x=(rand_x-0.5)*(0.16-2*size[0])
            rand_y=np.random.uniform()
            if rand_x>=0:
                rand_y=rand_y*(-(0.07-size[0]*np.sqrt(2))/(0.08-size[0])*rand_x-(-0.03-size[0]))
            else:
                rand_y=rand_y*((0.07-size[0]*np.sqrt(2))/(0.08-size[0])*rand_x-(-0.03-size[0]))
            
            
        else:
            theta=np.random.uniform(low=0,high=2*np.pi)
            r=np.random.uniform(low=0,high=size[0]/2)
            rand_x=np.sin(theta)*r
            rand_y=np.cos(theta)*r+0.02
        z = size[-1]
        
        return rand_x, rand_y, z    
    
    #Function to reset the simulator
    def reset(self,start_pos=None,obj_params=None,coords='global',qpos=None):
        # x, y = self.randomize_initial_pose(False, "s") # for RL training
        #x, y = self.randomize_initial_pose(True) # for data collection
        objects = self.experiment(1, 2)
    
        shapes=list(objects.keys())
        #print(shapes[0])
        #self._get_jacobian()
        
        hand_rotation=0#np.random.normal(-0.087,0.087,3)
        obj=0
        
        #hand_rotation=np.array([np.pi/2, 0, np.pi/2])
        
        #Might make sense to move this to experiments and use that to generate the objects and positions
        if obj_params !=None:
            if obj_params[0] == "Box":
                if obj_params[1] == "B":
                    obj=0
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
                elif obj_params[1] == "M":
                    obj=1
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
                elif obj_params[1] == "S":
                    obj=2
                    self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1.xml"
            elif obj_params[0] == "Cylinder":
                if obj_params[1] == "B":
                    obj=3
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"          
                elif obj_params[1] == "M":
                    obj=4
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
                elif obj_params[1] == "S":
                    obj=5
                    self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
            elif obj_params[0] == "Hour":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bhg.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bhg.xml"
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mhg.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mhg.xml"
                elif obj_params[1] == "S":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_shg.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_shg.xml"
            if obj_params[0] == "Vase":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bvase.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bvase.xml"
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mvase.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mvase.xml"
                elif obj_params[1] == "S":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_svase.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_svase.xml"
            elif obj_params[0] == "Bottle":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bbottle.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bbottle.xml"
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mbottle.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mbottle.xml"
                elif obj_params[1] == "S":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_sbottle.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_sbottle.xml"
            elif obj_params[0] == "Bowl":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bRoundBowl.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bRoundBowl.xml"
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mRoundBowl.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mRoundBowl.xml"
                elif obj_params[1] == "S":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_sRoundBowl.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_sRoundBowl.xml"
            if obj_params[0] == "Lemon":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_blemon.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_blemon.xml"
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mlemon.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mlemon.xml"
                elif obj_params[1] == "S":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_slemon.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_slemon.xml"
            elif obj_params[0] == "TBottle":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_btbottle.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_btbottle.xml"
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mtbottle.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mtbottle.xml"
                elif obj_params[1] == "S":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_stbottle.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_stbottle.xml"
            elif obj_params[0] == "RBowl":
                if obj_params[1] == "B":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bRectBowl.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bRectBowl.xml"
                elif obj_params[1] == "M":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mRectBowl.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mRectBowl.xml"
                elif obj_params[1] == "S":
                    self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_sRectBowl.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_sRectBowl.xml"

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
        #-1.57,0,-1.57 is side normal
        #-1.57, 0, 0 is side tilted
        #0,0,-1.57 is top down
        
        
        if self.filename=="/kinova_description/j2s7s300_end_effector.xml":
            new_rotation=np.array([0,0,0])+hand_rotation
            hand_orientation=np.random.rand()
            if hand_orientation <0.333:
                new_rotation=np.array([0,0,0])+hand_rotation
            elif hand_orientation >0.667:
                new_rotation=np.array([0,0,0])+hand_rotation
            else:
                new_rotation=np.array([1.57,0,0])+hand_rotation
        else:
            hand_orientation=0.1#np.random.rand()
            if hand_orientation <0.333:
                new_rotation=np.array([-1.57,0,-1.57])+hand_rotation
                #print('normal side grasp')
            elif hand_orientation >0.667:
                #print('top down grasp')
                new_rotation=np.array([0,0,0])+hand_rotation
            else:
                #print('tilted side grasp')
                new_rotation=np.array([-1.2,0,0])+hand_rotation
        
        
        #a,b,c=-hand_rotation[0],-hand_rotation[1],-hand_rotation[2]
        
        #new_motion=np.array([[np.cos(c)*np.cos(b), np.cos(c)*np.sin(b)*np.sin(a)-np.sin(c)*np.cos(a), np.cos(a)*np.sin(b)*np.cos(c)+np.sin(a)*np.sin(c)], \
        #                     [np.sin(c)*np.cos(b), np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c), np.sin(c)*np.sin(b)*np.cos(a)-np.cos(c)*np.sin(a)], \
        #                     [-np.sin(b), np.cos(b)*np.sin(a), np.cos(b)*np.cos(a)]])
        #Rx=np.array([[1,0,0],[0, np.cos(a), -np.sin(a)],[0, np.sin(a), np.cos(a)]])
        #Ry=np.array([[np.cos(b), 0, np.sin(b)],[0, 1, 0],[-np.sin(b), 0, np.cos(b)]])
        #Rz=np.array([[np.cos(c), -np.sin(c), 0],[np.sin(c), np.cos(c), 0],[0, 0, 1]])
        #new_motion=np.matmul(Rx,Ry)
        #new_motion=np.matmul(new_motion,Rz)
        #new_motion=np.matmul(new_motion,np.array([1,0,0]))
        #for i in range(3):
        #    if abs(new_motion[i]) <0.001:
        #        new_motion[i]=0
        #print('this is the motion,', new_motion)
        #print('this is the rotation,', new_rotation)
        xml_file=open(self.file_dir+self.filename,"r")
        xml_contents=xml_file.read()
        xml_file.close()
        #print('THIS IS THE CONTETNS', xml_contents)
        starting_point=xml_contents.find('<body name="j2s7s300_link_7"')
        euler_point=xml_contents.find('euler=',starting_point)
        #print('THIS IS THE POINT',euler_point)
        contents=re.search("[^\s]+\s[^\s]+\s[^>]+",xml_contents[euler_point:])
        c_start=contents.start()
        c_end=contents.end()
        
        starting_point=xml_contents.find('joint name="j2s7s300_joint_7" type')
        axis_point=xml_contents.find('axis=',starting_point)
        contents=re.search("[^\s]+\s[^\s]+\s[^>]+",xml_contents[axis_point:])
        #c2_start=contents.start()
        #c2_end=contents.end()
        #p4=str(new_motion[0])
        #p5=str(new_motion[1])
        #p6=str(new_motion[2])
        
        p1=str(new_rotation[0])
        p2=str(new_rotation[1])
        p3=str(new_rotation[2])
        

        
        xml_contents=xml_contents[:euler_point+c_start+7] + p1[0:min(5,len(p1))]+ " "+p2[0:min(5,len(p2))] +" "+ p3[0:min(5,len(p3))] \
        + xml_contents[euler_point+c_end-1:]#axis_point+c2_start+6] + p4[0:min(5,len(p4))]+ " "+p5[0:min(5,len(p5))] +" "+ p6[0:min(5,len(p6))] \
        #+ xml_contents[axis_point+c2_end-3:]
        #print(xml_contents)
        xml_file=open(self.file_dir+self.filename,"w")
        xml_file.write(xml_contents)
        xml_file.close()
        
        
        self._model = load_model_from_path(self.file_dir + self.filename)
        
        self._sim = MjSim(self._model)  
        self._set_state(np.array([0, 0, 0, 0, 0, 0, 10, 10, 10]))
        self._get_trans_mat_wrist_pose()
        if hand_orientation < 0.333:
            xloc,yloc,zloc,f1prox,f2prox,f3prox=0,0,0,0,0,0
        elif hand_orientation > 0.667:
            size=self._get_obj_size()
            stuff=np.matmul(self.Tfw[0:3,0:3],[0,-0.15,0.14+size[-1]])
            xloc,yloc,zloc,f1prox,f2prox,f3prox=-stuff[0],-stuff[1],stuff[2],0,0,0
        else:
            temp=np.matmul(self.Tfw[0:3,0:3],np.array([0,0,-0.06]))
            #print(temp)
            xloc,yloc,zloc,f1prox,f2prox,f3prox=temp[0],temp[1],temp[2],0,0,0
        if qpos is None:
            if start_pos is None:
                #print('hand orientation',hand_orientation)
                if hand_orientation >0.667:
                    x, y, z = self.randomize_initial_pos_data_collection(orientation='top')
                else:
                    x, y, z = self.randomize_initial_pos_data_collection(obj=obj)
            elif len(start_pos)==3:
                x, y, z = start_pos[0], start_pos[1], start_pos[2]
            elif len(start_pos)==2:
                x, y = start_pos[0], start_pos[1]
                z = self._get_obj_size()[-1]
            else:
                xloc,yloc,zloc,f1prox,f2prox,f3prox=start_pos[0], start_pos[1], start_pos[2],start_pos[3], start_pos[4], start_pos[5]
                x, y, z = start_pos[6], start_pos[7], start_pos[8]

            #all_states should be in the following format [xloc,yloc,zloc,f1prox,f2prox,f3prox,objx,objy,objz]
            self.all_states_1 = np.array([xloc, yloc, zloc, f1prox, f2prox, f3prox, x, y, z])
            if coords=='local':
                world_coords=np.matmul(self.Twf[0:3,0:3],np.array([x,y,z]))
                self.all_states_1=np.array([xloc, yloc, zloc, f1prox, f2prox, f3prox, world_coords[0], world_coords[1], world_coords[2]])
            self.Grasp_Reward=False
            self.all_states_2 = np.array([xloc, yloc, zloc+0.03, f1prox, f2prox, f3prox, 0.05, 0.0, 0.055])
            self.all_states = [self.all_states_1 , self.all_states_2] 
    
            self._set_state(self.all_states[1])
        else:
            self.set_sim_state(qpos,start_pos)
            x, y, z = start_pos[0], start_pos[1], start_pos[2]
        #TODO: fix this so that we can still put hourglasses where they are supposed to go
        states = self._get_obs()
        obj_pose=self._get_obj_pose()
        #obj_pose=np.array(obj_pose)
        #obj_pose=np.append(obj_pose,1)
        #obj_pose=self.Tfw*obj_pose
        deltas=[x-obj_pose[0],y-obj_pose[1],z-obj_pose[2]]

        
        if np.linalg.norm(deltas)>0.05:
            self.all_states_1=np.array([xloc, yloc, zloc, f1prox, f2prox, f3prox, x+deltas[0], y+deltas[1], z+deltas[2]])
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
    def step(self, action): #TODO: fix this so that we can rotate the hand
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
        
        if self.arm_or_hand=="hand":
            mass=0.732
            gear=25
            stuff=np.matmul(self.Tfw[0:3,0:3],[0,0,mass*10/gear])
            stuff[0]=-stuff[0]
            stuff[1]=-stuff[1]
            for _ in range(self.frame_skip):
                #print('tfw',self.Tfw)
                if self.step_coords=='global':
                    slide_vector=np.matmul(self.Tfw[0:3,0:3],action[0:3])
                    slide_vector=[-slide_vector[0],-slide_vector[1],slide_vector[2]]
                else:
                    slide_vector=[-action[0],-action[1],action[2]]#np.matmul(self.Twf[0:3,0:3],action[0:3])
                #print(self.Twf[0:3,0:3])
                '''
                if action[0] < 0.0:
                    self._sim.data.ctrl[0] = 0.0
                else:    
                    #self._sim.data.ctrl[0] = (action[0] / 0.8) * 0.2
                    self._sim.data.ctrl[0] = action[0]
                '''
                if len(action) ==4:
                    self._sim.data.ctrl[0] = action[0]
                    self._sim.data.ctrl[1]=stuff[2]
                    for i in range(3):
                        self._sim.data.ctrl[i+2] = action[i+1]
                else:
                    for i in range(3):
                        self._sim.data.ctrl[(i)*2] = slide_vector[i]
                        self._sim.data.ctrl[i+6] = action[i+3]
                        self._sim.data.ctrl[i*2+1]=stuff[i]
                self._sim.step()
            #print('slide vector',slide_vector)
        else:
            for _ in range(self.frame_skip):
                joint_velocities = action[0:7]
                finger_velocities=action[7:]
                '''
                if action[0] < 0.0:
                    self._sim.data.ctrl[0] = 0.0
                else:    
                    #self._sim.data.ctrl[0] = (action[0] / 0.8) * 0.2
                    self._sim.data.ctrl[0] = action[0]
                '''
                
                for i in range(len(joint_velocities)):
                    # vel = action[i]
                    
                    #if action[i+1] < 0.0:
                    #self._sim.data.ctrl[i+1] = 0.0
                        
                    
                    #else:    
                    #self._sim.data.ctrl[(i)*2+1] = -tot_external_force[i]/24
                    self._sim.data.ctrl[i+10] = joint_velocities[i]
                for i in range(len(finger_velocities)):
                    self._sim.data.ctrl[i+7] = finger_velocities[i]
                #print('final ctrl stuff', self._sim.data.ctrl)
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
