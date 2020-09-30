
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
import csv
import pandas as pd
import threading #oh boy this might get messy

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
            self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mbox.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bbox.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bbox.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_shg.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_shg.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mhg.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mhg.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bhg.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bhg.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_svase.xml"), 's',"/kinova_description/j2s7s300_end_effector_v1_svase.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mvase.xml"), 'm',"/kinova_description/j2s7s300_end_effector_v1_mvase.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bvase.xml"), 'b',"/kinova_description/j2s7s300_end_effector_v1_bvase.xml"
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
            #self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/DisplayStuff.xml"),'s',"/kinova_description/DisplayStuff.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcone1.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bcone1.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcone1.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mcone1.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scone1.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1_scone1.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcone2.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bcone2.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcone2.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mcone2.xml"
            #self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scone2.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1_scone2.xml"

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
        self.pid=False
        self.step_coords='global'
        self._torque = [0,0,0,0] #Unused
        self._velocity = [0,0,0,0] #Unused
        self._jointAngle = [5,0,0,0] #Usused
        self._positions = [] # ??
        self._numSteps = 0
        self._simulator = "Mujoco"
        self.action_scale = 0.0333
        self.max_episode_steps = 150
        self.site_count=0
        # Parameters for cost function
        self.state_des = 0.20
        self.initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.action_space = spaces.Box(low=np.array([-0.8, -0.8, -0.8, -0.8]), high=np.array([0.8, 0.8, 0.8, 0.8]), dtype=np.float32) # Velocity action space
        self.const_T=np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])  #Transfer matrix from world frame to un-modified hand frame
        self.frame_skip = frame_skip # Used in step. Number of frames you go through before you reach the next step
        self.all_states = None  # This is the varriable we use to save the states before they are sent to the simulator when we are resetting.

        self.state_rep = "local" # change accordingly

        self.obj_coords = [0,0,0]
        self.objects = {}
        self.obj_keys = list()

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


        #self.Grasp_net = LinearNetwork().to(device) # This loads the grasp classifier
        #trained_model = "/home/orochi/KinovaGrasping/gym-kinova-gripper/trained_model_05_28_20_2105local.pt"
        #trained_model = "/home/orochi/KinovaGrasping/gym-kinova-gripper/trained_model_01_23_20_2052local.pt"
        # self.Grasp_net = GraspValid_net(54).to(device)
        # trained_model = "/home/graspinglab/NCS_data/ExpertTrainedNet_01_04_20_0250.pt"
        #model = torch.load(trained_model)
        #self.Grasp_net.load_state_dict(model)
        #self.Grasp_net.eval()


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
        temp=np.matmul(Rfa,np.array([[0,0,1],[-1,0,0],[0,-1,0]]))
        temp=np.transpose(temp)
        Tfa=np.zeros([4,4])
        Tfa[0:3,0:3]=temp
        Tfa[3,3]=1       
        Tfw=np.zeros([4,4])
        Tfw[0:3,0:3]=temp
        Tfw[3,3]=1
        self.wrist_pose=self.wrist_pose+np.matmul(np.transpose(Tfw[0:3,0:3]),[-0.009,0.048,0.0])
        Tfw[0:3,3]=np.matmul(-(Tfw[0:3,0:3]),np.transpose(self.wrist_pose))
        self.Tfw=Tfw 
        self.Twf=np.linalg.inv(Tfw)

    def experimental_sensor(self,rangedata,finger_pose,gravity):
        #print('flimflam')
        #finger_joints = ["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
        finger_pose=np.array(finger_pose)

        s1=finger_pose[0:3]-finger_pose[6:9]
        s2=finger_pose[0:3]-finger_pose[3:6]
        #print(finger_pose)
        front_area=np.linalg.norm(np.cross(s1,s2))/2
        #print('front area',front_area)
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

        return sensor_pose,front_part, top_part


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


    def obs_test(self):
        obj_pose = self._get_obj_pose()
        obj_pose = np.copy(obj_pose)
        tests_passed=[]
        self._sim.data.site_xpos[0]=obj_pose
        self._sim.data.site_xpos[1]=obj_pose
        print(self._sim.data.qpos)
        print('object position', obj_pose)
        temp=True
        while temp:
            ans=input('do the red bars line up with the object center Y/N?')
            if ans.lower()=='n':
                print('Recording first test as failure')
                tests_passed.append(False)
                temp=False
            elif ans.lower()=='y':
                print('Recording first test as success')
                tests_passed.append(True)
                temp=False
            else:
                print('input not recognized, please input either Y or N. do the red bars line up with the object center Y/N?')
        print('Next test, finger positions')
            
        finger_joints = ["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
        fingers_6D_pose = []
        for joint in finger_joints:
                trans = self._sim.data.get_geom_xpos(joint)
                trans = list(trans)
                for i in range(3):
                    fingers_6D_pose.append(trans[i])
        for i in range(6):
            self._sim.data.site_xpos[0]=fingers_6D_pose[i*3:i*3+3]
            self._sim.data.site_xpos[1]=fingers_6D_pose[i*3:i*3+3]
            temp=True
            while temp:
                ans=input(f'do the red bars line up with the {finger_joints[i]} Y/N?')
                if ans.lower()=='n':
                    print('Recording test as failure')
                    tests_passed.append(False)
                    temp=False
                elif ans.lower()=='y':
                    print('Recording test as success')
                    tests_passed.append(True)
                    temp=False
                else:
                    print(f'input not recognized, please input either Y or N. do the red bars line up with the {finger_joints[i]} Y/N?')
        print('Next test, wrist position')
        self._sim.data.site_xpos[0]=self.wrist_pose
        self._sim.data.site_xpos[1]=self.wrist_pose
        temp=True
        while temp:
            ans=input('do the red bars line up with the wrist position Y/N?')
            if ans.lower()=='n':
                print('Recording first test as failure')
                tests_passed.append(False)
                temp=False
            elif ans.lower()=='y':
                print('Recording first test as success')
                tests_passed.append(True)
                temp=False
            else:
                print('input not recognized, please input either Y or N. do the red bars line up with the wrist position Y/N?')
        passed=np.sum(tests_passed)
        failed=np.sum(np.invert(tests_passed))
        print('out of', np.shape(tests_passed), f'tests, {passed} tests passed and {failed} tests failed')
        print('tests passed')
        print('object pose:',tests_passed[0])
        print('wrist pose:',tests_passed[7])
        for i in range(6):
            print(finger_joints[i], 'pose:',tests_passed[i+1])

        
            
        
    # Function to return global or local transformation matrix
    def _get_obs(self, state_rep=None, test = False):  #TODO: Add or subtract elements of this to match the discussions with Ravi and Cindy
        '''
        Local obs, all in local coordinates (from the center of the palm)
        (18,) Finger Pos										0-18
        (3,) Wrist Pos											18-21
        (3,) Obj Pos											21-24
        (9,) Joint States										24-33
        (3,) Obj Size											33-36
        (12,) Finger Object Distance							36-48
        (2,) X and Z angle										48-50
        (17,) Rangefinder data									50-67
        (3,) Gravity vector in local coordinates					67-70
        (3,) Object location based on rangefinder data				70-73
        (1,) Ratio of the area of the side of the shape to the open portion of the side of the hand	73
        (1,) Ratio of the area of the top of the shape to the open portion of the top of the hand	74
        '''
        '''
        Global obs, all in global coordinates (from simulator 0,0,0)
        (18,) Finger Pos										0-18
        (3,) Wrist Pos											18-21
        (3,) Obj Pos											21-24
        (9,) Joint States										24-33
        (3,) Obj Size											33-36
        (12,) Finger Object Distance							36-48
        (2,) X and Z angle										48-50
        (17,) Rangefinder data									50-67
        '''
        
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
        if test:
            x=threading.Thread(target=self.obs_test)
            x.start()
            while x.is_alive():
                self.render()
            print('')
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
            wrist_for_rotation=np.append(self.wrist_pose,1)
            wrist_for_rotation=np.matmul(self.Tfw,wrist_for_rotation)

            wrist_pose = wrist_for_rotation[0:3]
            obj_for_roation=np.append(obj_pose,1)
            obj_for_roation=np.matmul(self.Tfw,obj_for_roation)
            obj_pose = obj_for_roation[0:3]
            gravity=np.matmul(self.Tfw[0:3,0:3],gravity)
            sensor_pos,front_thing,top_thing=self.experimental_sensor(range_data,fingers_6D_pose,gravity)
            fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [x_angle, z_angle] + range_data + [gravity[0],gravity[1],gravity[2]] + [sensor_pos[0],sensor_pos[1],sensor_pos[2]] + [front_thing, top_thing] #+ [self.obj_shape]
            if self.pid:
                fingers_6D_pose = fingers_6D_pose+ [self._get_dot_product()]
        elif state_rep == "joint_states":
            fingers_6D_pose = joint_states + list(obj_pose) + [obj_size[0], obj_size[1], obj_size[2]*2] + [x_angle, z_angle] #+ fingers_dot_prod
        return fingers_6D_pose

    # Function to get the distance between the digits on the fingers and the object center
    # NOTE! This only takes into account the x and y differences. We might want to consider taking z into account as well for other orientations
    def _get_finger_obj_dist(self): 
        finger_joints = ["f1_prox", "f1_prox_1", "f2_prox", "f2_prox_1", "f3_prox", "f3_prox_1","f1_dist", "f1_dist_1", "f2_dist", "f2_dist_1", "f3_dist", "f3_dist_1"]

        obj = self._get_obj_pose()
        dists = []
        for i in finger_joints:
            pos = self._sim.data.get_site_xpos(i)
            dist = np.absolute(pos[0:3] - obj[0:3])
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
        return x_angle,z_angle
    
    #function to get the dot product. Only used for the pid controller
    def _get_dot_product(self):
        obj_state = self._get_obj_pose()
        hand_pose = self._sim.data.get_body_xpos("j2s7s300_link_7")
        obj_state_x = abs(obj_state[0] - hand_pose[0])
        obj_state_y = abs(obj_state[1] - hand_pose[1])
        obj_vec = np.array([obj_state_x, obj_state_y])
        obj_vec_norm = np.linalg.norm(obj_vec)
        obj_unit_vec = obj_vec / obj_vec_norm

        center_x = abs(0.0 - hand_pose[0])
        center_y = abs(0.0 - hand_pose[1])
        center_vec = np.array([center_x, center_y])
        center_vec_norm = np.linalg.norm(center_vec)
        center_unit_vec = center_vec / center_vec_norm
        dot_prod = np.dot(obj_unit_vec, center_unit_vec)
        return dot_prod**20 # cuspy to get distinct reward


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

        info = {"lift_reward":lift_reward}
        return lift_reward, info, done

    # Function to get rewards for RL training
    def _get_reward(self, test = False): # TODO: change obs[23] and obs[5] to the simulator height object and stop using _get_obs
        #TODO: Make sure this works with the new grasp classifier

        # object height target
        obj_target = 0.2

        # Grasp reward
        grasp_reward = 0.0
        obs = self._get_obs(state_rep="global")
        loc_obs=self._get_obs()

        # network_inputs=obs[0:5]
        # network_inputs=np.append(network_inputs,obs[6:23])
        # network_inputs=np.append(network_inputs,obs[24:])
        # inputs = torch.FloatTensor(np.array(network_inputs)).to(device)

        # WITHOUT GRASP CLASSIFIER
        #if np.max(np.array(obs[41:47])) < 0.035 or np.max(np.array(obs[35:41])) < 0.015:
        #     outputs = self.Grasp_net(inputs).cpu().data.numpy().flatten()
        #     if (outputs >=0.3) & (not self.Grasp_Reward):
        #         grasp_reward = 5.0
        #         self.Grasp_Reward=True
        #     else:
        #         grasp_reward = 0.0

        if not test:
            if abs(obs[23] - obj_target) < 0.005 or (obs[23] >= obj_target):
                lift_reward = 50.0
                done = True
            else:
                lift_reward = 0.0
                done = False

            finger_reward = -np.sum((np.array(obs[41:47])) + (np.array(obs[35:41])))

        else:
            lift_reward = 0.0
            done = False
            finger_reward = -12.0

        #finger_reward = -np.sum((np.array(obs[41:47])) + (np.array(obs[35:41])))

        reward = 0.2*finger_reward + lift_reward + grasp_reward
        info = {"lift_reward":lift_reward}

        if test:
            if reward == -2.4000000000000004:
                print("Reward function Working Properly")
            else:
                print("Reward function is not Working Properly")

        return reward, info, done

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
        #print(self._sim.model.geom_size)
        #print(num_of_geoms[0]-8)
        for i in range(num_of_geoms[0]-8):
            size=np.copy(self._sim.model.geom_size[-1-i])
            diffs=[0,0,0]
            if size[2]==0:
                size[2]=size[1]
                size[1]=size[0]
            diffs[0]=abs(size[0]-size[1])
            diffs[1]=abs(size[1]-size[2])
            diffs[2]=abs(size[0]-size[2])
            if ('lemon' in self.filename)|(np.argmin(diffs)!=0):
                temp=size[0]
                size[0]=size[2]
                size[2]=temp

            if 'Bowl' in self.filename:
                if 'Rect' in self.filename:
                    final_size[0]=0.17
                    final_size[1]=0.17
                    final_size[2]=0.075
                else:
                    final_size[0]=0.175
                    final_size[1]=0.175
                    final_size[2]=0.07
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
        #print(final_size)
        return final_size

    def set_obj_coords(self,x,y,z):
        self.obj_coords[0] = x
        self.obj_coords[1] = y
        self.obj_coords[2] = z

    def get_obj_coords(self):
        return self.obj_coords

    # Function to run all the experiments for RL training
    def experiment(self, shape_keys): #TODO: Talk to people thursday about adding the hourglass and bottles to this dataset.
        all_objects = {}
        # Cube
        all_objects["CubeS"] = "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"
        all_objects["CubeM"] = "/kinova_description/j2s7s300_end_effector_v1_CubeM.xml"
        all_objects["CubeB"] = "/kinova_description/j2s7s300_end_effector_v1_CubeB.xml"
        # Cylinder
        all_objects["CylinderS"] = "/kinova_description/j2s7s300_end_effector_v1_CylinderS.xml"
        all_objects["CylinderM"] = "/kinova_description/j2s7s300_end_effector_v1_CylinderM.xml"
        all_objects["CylinderB"] = "/kinova_description/j2s7s300_end_effector_v1_CylinderB.xml"
        # Cube rotated by 45 degrees
        all_objects["Cube45S"] = "/kinova_description/j2s7s300_end_effector_v1_Cube45S.xml"
        all_objects["Cube45M"] = "/kinova_description/j2s7s300_end_effector_v1_Cube45M.xml"
        all_objects["Cube45B"] = "/kinova_description/j2s7s300_end_effector_v1_Cube45B.xml"
        # Vase 1
        all_objects["Vase1S"] = "/kinova_description/j2s7s300_end_effector_v1_Vase1S.xml"
        all_objects["Vase1M"] = "/kinova_description/j2s7s300_end_effector_v1_Vase1M.xml"
        all_objects["Vase1B"] = "/kinova_description/j2s7s300_end_effector_v1_Vase1B.xml"
        # Vase 2
        all_objects["Vase2S"] = "/kinova_description/j2s7s300_end_effector_v1_Vase2S.xml"
        all_objects["Vase2M"] = "/kinova_description/j2s7s300_end_effector_v1_Vase2M.xml"
        all_objects["Vase2B"] = "/kinova_description/j2s7s300_end_effector_v1_Vase2B.xml"
        # Cone 1
        all_objects["Cone1S"] = "/kinova_description/j2s7s300_end_effector_v1_Cone1S.xml"
        all_objects["Cone1M"] = "/kinova_description/j2s7s300_end_effector_v1_Cone1M.xml"
        all_objects["Cone1B"] = "/kinova_description/j2s7s300_end_effector_v1_Cone1B.xml"
        # Cone 2
        all_objects["Cone2S"] = "/kinova_description/j2s7s300_end_effector_v1_Cone2S.xml"
        all_objects["Cone2M"] = "/kinova_description/j2s7s300_end_effector_v1_Cone2M.xml"
        all_objects["Cone2B"] = "/kinova_description/j2s7s300_end_effector_v1_Cone2B.xml"

        ## Nigel's Shapes ##
        # Hourglass
        all_objects["HourB"] =  "/kinova_description/j2s7s300_end_effector_v1_bhg.xml"
        all_objects["HourM"] =  "/kinova_description/j2s7s300_end_effector_v1_mhg.xml"
        all_objects["HourS"] =  "/kinova_description/j2s7s300_end_effector_v1_shg.xml"
        # Vase
        all_objects["VaseB"] =  "/kinova_description/j2s7s300_end_effector_v1_bvase.xml"
        all_objects["VaseM"] =  "/kinova_description/j2s7s300_end_effector_v1_mvase.xml"
        all_objects["VaseS"] =  "/kinova_description/j2s7s300_end_effector_v1_svase.xml"
        # Bottle
        all_objects["BottleB"] =  "/kinova_description/j2s7s300_end_effector_v1_bbottle.xml"
        all_objects["BottleM"] =  "/kinova_description/j2s7s300_end_effector_v1_mbottle.xml"
        all_objects["BottleS"] =  "/kinova_description/j2s7s300_end_effector_v1_sbottle.xml"
        # Bowl
        all_objects["BowlB"] =  "/kinova_description/j2s7s300_end_effector_v1_bRoundBowl.xml"
        all_objects["BowlM"] =  "/kinova_description/j2s7s300_end_effector_v1_mRoundBowl.xml"
        all_objects["BowlS"] =  "/kinova_description/j2s7s300_end_effector_v1_sRoundBowl.xml"
        # Lemon
        all_objects["LemonB"] =  "/kinova_description/j2s7s300_end_effector_v1_blemon.xml"
        all_objects["LemonM"] =  "/kinova_description/j2s7s300_end_effector_v1_mlemon.xml"
        all_objects["LemonS"] =  "/kinova_description/j2s7s300_end_effector_v1_slemon.xml"
        # TBottle
        all_objects["TBottleB"] =  "/kinova_description/j2s7s300_end_effector_v1_btbottle.xml"
        all_objects["TBottleM"] =  "/kinova_description/j2s7s300_end_effector_v1_mtbottle.xml"
        all_objects["TBottleS"] =  "/kinova_description/j2s7s300_end_effector_v1_stbottle.xml"
        # RBowl
        all_objects["RBowlB"] =  "/kinova_description/j2s7s300_end_effector_v1_bRectBowl.xml"
        all_objects["RBowlM"] =  "/kinova_description/j2s7s300_end_effector_v1_mRectBowl.xml"
        all_objects["RBowlS"] =  "/kinova_description/j2s7s300_end_effector_v1_sRectBowl.xml"

        for key in shape_keys:
            self.objects[key] = all_objects[key]

        if len(shape_keys) == 0:
            print("No shape keys")
            raise ValueError
        elif len(shape_keys) != len(self.objects):
            print("Invlaid shape key requested")
            raise ValueError
        return self.objects

    #Function to randomize the position of the object for grasp classifier data collection
    def randomize_initial_pos_data_collection(self,orientation="side"):
        print('ya done messed up A-A-ron')
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
        elif orientation=='rotated':
            rand_x=0
            rand_y=0
        else:
            theta=np.random.uniform(low=0,high=2*np.pi)
            r=np.random.uniform(low=0,high=size[0]/2)
            rand_x=np.sin(theta)*r
            rand_y=np.cos(theta)*r
        z = size[-1]/2
        return rand_x, rand_y, z 

    def write_xml(self,new_rotation):   #This function takes in a rotation vector [roll, pitch, yaw] and sets the hand rotation in the
                                        #self.file_dir and self.filename to that rotation. It then sets up the simulator with the object 
                                        #incredibly far from the hand to prevent collisions and recalculates the rotation matrices of the hand
        xml_file=open(self.file_dir+self.filename,"r")
        xml_contents=xml_file.read()
        xml_file.close()
        starting_point=xml_contents.find('<body name="j2s7s300_link_7"')
        euler_point=xml_contents.find('euler=',starting_point)
        contents=re.search("[^\s]+\s[^\s]+\s[^>]+",xml_contents[euler_point:])
        c_start=contents.start()
        c_end=contents.end()
        starting_point=xml_contents.find('joint name="j2s7s300_joint_7" type')
        axis_point=xml_contents.find('axis=',starting_point)
        contents=re.search("[^\s]+\s[^\s]+\s[^>]+",xml_contents[axis_point:])
        starting_point=xml_contents.find('site name="local_origin_site" type="cylinder" size="0.0075 0.005" rgba="25 0.5 0.0 1"')
        site_point=xml_contents.find('pos=',starting_point)
        contents=re.search("[^\s]+\s[^\s]+\s[^>]+",xml_contents[starting_point:])
        wrist_pose=self.wrist_pose
        new_thing= str(wrist_pose[0]) + " " + str(wrist_pose[1]) + " " + str(wrist_pose[2])
        p1=str(new_rotation[0])
        p2=str(new_rotation[1])
        p3=str(new_rotation[2])
        xml_contents=xml_contents[:euler_point+c_start+7] + p1[0:min(5,len(p1))]+ " "+p2[0:min(5,len(p2))] +" "+ p3[0:min(5,len(p3))] \
        + xml_contents[euler_point+c_end-1:]# + new_thing + xml_contents[site_point+c2_end:]
        xml_file=open(self.file_dir+self.filename,"w")
        xml_file.write(xml_contents)
        xml_file.close()
        self._model = load_model_from_path(self.file_dir + self.filename)
        self._sim = MjSim(self._model) 
        self._set_state(np.array([0, 0, 0, 0, 0, 0, 10, 10, 10]))
        self._get_trans_mat_wrist_pose()
        
    # Steph Added
    def check_obj_file_empty(self,filename):
        if os.path.exists(filename) == False:
            return False
        with open(filename, 'r') as read_obj:
            # read first character
            one_char = read_obj.read(1)
            # if not fetched then file is empty
            if not one_char:
               return True
            return False

    def Generate_Latin_Square(self,max_elements,filename,shape_keys, test = False):
        print("GENERATE LATIN SQUARE")
        ### Choose an experiment ###
        self.objects = self.experiment(shape_keys)

        # n is the number of object types (sbox, bbox, bcyl, etc.)
        num_elements = 0
        elem_gen_done = 0
        printed_row = 0

        while num_elements < max_elements:
            n = len(self.objects.keys())-1
            #print("This is n: ",n)
            k = n
            # Loop to prrows
            for i in range(0, n+1, 1):
                # This loops runs only after first iteration of outer loop
                # Prints nummbers from n to k
                keys = list(self.objects.keys())
                temp = k

                while (temp <= n) :
                    if printed_row <= n: # Just used to print out one row instead of all of them
                        printed_row += 1

                    key_name = str(keys[temp])
                    self.obj_keys.append(key_name)
                    temp += 1
                    num_elements +=1
                    if num_elements == max_elements:
                        elem_gen_done = 1
                        break
                if elem_gen_done:
                    break

                # This loop prints numbers from 1 to k-1.
                for j in range(0, k):
                    key_name = str(keys[j])
                    self.obj_keys.append(key_name)
                    num_elements +=1
                    if num_elements == max_elements:
                        elem_gen_done = 1
                        break
                if elem_gen_done:
                    break
                k -= 1

########## Function Testing Code########
            if test:
                test_key = self.obj_keys
                if len(test_key) == max_elements:
                    test_key.sort()
                    num_elem_test = 1
                    for i in range(len(test_key)-2):
                        if test_key[i] != test_key[i+1]:
                            num_elem_test += 1

                    if num_elem_test == len(shape_keys):
                        print("Latin Square function is Generating Perfect Distribution")
                    else:
                        print("Latin Square function is not Generating Perfect Distribution")
########## Ends Here ###############

            w = csv.writer(open(filename, "w"))
            for key in self.obj_keys:
                w.writerow(key)

    def objects_file_to_list(self,filename, num_objects,shape_keys):
        print("FILENAME: ",filename)
        #print('FIRST OBJECT KEYS',self.obj_keys)
        df = pd.read_csv(filename,header=None, sep='\n')
        if (df.empty):
            "Object file is empty!"
            self.Generate_Latin_Square(num_objects,filename,shape_keys)
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row = ''.join(row)
                self.obj_keys.append(row)
        #print('LAST OBJECT KEYS',self.obj_keys)
    def get_obj_keys(self):
        return self.obj_keys

    def get_object(self,filename):
        # Get random shape
        random_shape = self.obj_keys.pop()

        # remove current object file contents
        f = open(filename, "w")
        f.truncate()
        f.close()

        # write new object keys to file so new env will have updated list
        w = csv.writer(open(filename, "w"))
        for key in self.obj_keys:
            w.writerow(key)

        # Load model
        self._model = load_model_from_path(self.file_dir + self.objects[random_shape])
        self._sim = MjSim(self._model)

        print("random_shape: ",random_shape)

        return random_shape, self.objects[random_shape]

    # Get the initial object position
    def sample_initial_valid_object_pos(self,shapeName,coords_filename):
        with open(coords_filename) as csvfile:
            checker=csvfile.readline()
            if ',' in checker:
                delim=','
            else:
                delim=' '
            for i in csv.reader(csvfile, delimiter= delim):
                data = [(float(i[0]), float(i[1]), float(i[2]))]
        rand_coord = random.choice(data)
        x = rand_coord[0]
        y = rand_coord[1]
        z = rand_coord[2]

        return x, y, z

    def obj_shape_generator(self,obj_params):
        if obj_params[0] == "Cube":
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
        elif obj_params[0] == "Cone1":
            if obj_params[1] == "B":
                self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcone1.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bcone1.xml"
            elif obj_params[1] == "M":
                self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcone1.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mcone1.xml"
            elif obj_params[1] == "S":
                self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scone1.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1_scone1.xml"
        elif obj_params[0] == "Cone2":
            if obj_params[1] == "B":
                self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcone2.xml"),'b',"/kinova_description/j2s7s300_end_effector_v1_bcone2.xml"
            elif obj_params[1] == "M":
                self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcone2.xml"),'m',"/kinova_description/j2s7s300_end_effector_v1_mcone2.xml"
            elif obj_params[1] == "S":
                self._model,self.obj_size,self.filename= load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scone2.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1_scone2.xml"
        elif obj_params[0]=='display':
            self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/DisplayStuff.xml"),'s',"/kinova_description/DisplayStuff.xml"
        return obj_params[0]+obj_params[1]

    def reset(self,env_name="env",shape_keys=["CubeS","CubeB","CylinderS","CylinderB","Cube45S","Cube45B","Cone1S","Cone1B","Cone2S","Cone2B","Vase1S","Vase1B","Vase2S","Vase2B"],hand_orientation="random",mode="train",start_pos=None,obj_params=None,coords='global',qpos=None, test = False):
        # x, y = self.randomize_initial_pose(False, "s") # for RL training
        #x, y = self.randomize_initial_pose(True) # for data collection

        print("Within Reset, hand_orientation: ",hand_orientation)

        # Steph new code
        obj_list_filename = ""
        num_objects = 200
        if env_name == "env":
            obj_list_filename = "objects.csv"
            num_objects = 20000
        else:
            obj_list_filename = "eval_objects.csv"
            num_objects = 200

        if len(self.objects) == 0:
            self.objects = self.experiment(shape_keys)
        if len(self.obj_keys) == 0:
            self.objects_file_to_list(obj_list_filename,num_objects,shape_keys)
            
        if obj_params==None:
            random_shape, self.filename = self.get_object(obj_list_filename)
        else:
            random_shape = self.obj_shape_generator(obj_params)

        #coords_filename = "gym_kinova_gripper/envs/kinova_description/shape_coords/" + random_shape + ".txt"

        # End of stephs new code

        shapes=list(self.objects.keys())
        #print(shapes[0])
        #self._get_jacobian()

        hand_rotation= 0 #np.random.normal(-0.087,0.087,3)
        obj=0

        # Orientation is initialized as Normal
        orientation_type = 0.330

        #-1.57,0,-1.57 is side normal
        #-1.57, 0, 0 is side tilted
        #0,0,-1.57 is top down


        if self.filename=="/kinova_description/j2s7s300_end_effector.xml":
            new_rotation=np.array([0,0,0])+hand_rotation
            if hand_orientation == 'random':
                orientation_type=np.random.rand()
                print("orientation_type: ",orientation_type)
                if orientation_type <0.333:
                    new_rotation=np.array([0,0,0])+hand_rotation
                elif orientation_type >0.667:
                    new_rotation=np.array([0,0,0])+hand_rotation
                else:
                    new_rotation=np.array([1.2,0,0])+hand_rotation
            else:
                # Normal hand orientation
                new_rotation=np.array([0,0,0])+hand_rotation

        else:
            if hand_orientation == 'random':
                if not test:
                    orientation_type=np.random.rand()
                else:
                    orientation_type= 0.111
                print("orientation_type: ",orientation_type)

                # Initial position
                if orientation_type <0.333:
                    new_rotation=np.array([-1.57,0,-1.57])+hand_rotation
                    coords_filename = "gym_kinova_gripper/envs/kinova_description/"+mode+"_coords/Normal/" + random_shape + ".txt"
                # Top orientation
                elif orientation_type >0.667:
                    new_rotation=np.array([0,0,0])+hand_rotation
                    coords_filename = "gym_kinova_gripper/envs/kinova_description/"+mode+"_coords/Top/" + random_shape + ".txt"
                # Sideways orientation
                else:
                    new_rotation=np.array([-1.2,0,0])+hand_rotation
                    coords_filename = "gym_kinova_gripper/envs/kinova_description/"+mode+"_coords/Side/" + random_shape + ".txt"
            else:
                # Normal hand orientations
                new_rotation=np.array([-1.57,0,-1.57])+hand_rotation
                coords_filename = "gym_kinova_gripper/envs/kinova_description/"+mode+"_coords/Normal/" + random_shape + ".txt"
        print("COORDS FILENAME: ",coords_filename)
        if test & np.shape(self._sim.data.site_xpos)[0]<19:
            self.add_site([0,0,0])
        self.write_xml(new_rotation)

        if orientation_type < 0.333:
            self.orientation = 'normal'
            xloc,yloc,zloc,f1prox,f2prox,f3prox=0,0,0,0,0,0
        elif orientation_type > 0.667:
            self.orientation = 'top'
            size=self._get_obj_size()
            
            if self.obj_size=='b':
                Z=0.15
            elif self.obj_size=='m':
                Z=0.14
            elif self.obj_size=='s':
                Z=0.13
            stuff=np.matmul(self.Tfw[0:3,0:3],[-0.005,-0.155,Z+0.06])
            #stuff=np.matmul(self.Tfw[0:3,0:3],[0,-0.15,0.1+size[-1]*1.8])
            xloc,yloc,zloc,f1prox,f2prox,f3prox=-stuff[0],-stuff[1],stuff[2],0,0,0
        else:
            self.orientation = 'rotated'
            temp=np.matmul(self.Tfw[0:3,0:3],np.array([0.051,-0.075,0.06]))
            #print('temp',temp)
            xloc,yloc,zloc,f1prox,f2prox,f3prox=-temp[0],-temp[1],temp[2],0,0,0
        if qpos is None:
            if start_pos is None:
                if orientation_type >0.667:
                  # Check for coords text file
                  if self.check_obj_file_empty(coords_filename) == False:
                      x, y, z = self.sample_initial_valid_object_pos(random_shape,coords_filename)
                      print("YOU CHOSE sample_initial_valid_object_pos: ",x,",",y,",",z)
                  else:
                      x, y, z = self.randomize_initial_pos_data_collection(orientation='top')
                      print("YOU CHOSE sample_initial_valid_object_pos: ",x,",",y,",",z)
                else:
                  if self.check_obj_file_empty(coords_filename) == False:
                      x, y, z = self.sample_initial_valid_object_pos(random_shape,coords_filename)
                  else:
                      x, y, z = self.randomize_initial_pos_data_collection()
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
            #if coords=='local':
            #    world_coords=np.matmul(self.Twf[0:3,0:3],np.array([x,y,z]))
            #    self.all_states_1=np.array([xloc, yloc, zloc, f1prox, f2prox, f3prox, world_coords[0], world_coords[1], world_coords[2]])
            self.Grasp_Reward=False
            self.all_states_2 = np.array([xloc, yloc, zloc, f1prox, f2prox, f3prox, 0.00, 0.0, 0.055])
            self.all_states = [self.all_states_1 , self.all_states_2]

            self._set_state(self.all_states[0])
        else:
            self.set_sim_state(qpos,start_pos)
            x, y, z = start_pos[0], start_pos[1], start_pos[2]
        states = self._get_obs(test=False)
        obj_pose=self._get_obj_pose()
        deltas=[x-obj_pose[0],y-obj_pose[1],z-obj_pose[2]]
        #print('deltas',deltas)

        if np.linalg.norm(deltas)>0.05:
            self.all_states_1=np.array([xloc, yloc, zloc, f1prox, f2prox, f3prox, x+deltas[0], y+deltas[1], z+deltas[2]])
            self.all_states=[self.all_states_1,self.all_states_2]
            self._set_state(self.all_states[0])
            states = self._get_obs(test=False)

        #These two varriables are used when the action space is in joint states
        self.t_vel = 0
        self.prev_obs = []
        
        # Sets the object coordinates for heatmap tracking and plotting
        self.set_obj_coords(x,y,z)
        self._get_trans_mat_wrist_pose()


        ##Testing Code
        if test:
            if [xloc, yloc, zloc, f1prox, f2prox, f3prox] == [0,0,0,0,0,0]: 
                if coords_filename == "gym_kinova_gripper/envs/kinova_description/"+mode+"_coords/Normal/" + random_shape + ".txt":
                    print("Reset function is working Properly Check the render")
                    self.render()
            else:
                print("Reset function is not working Properly Check the render")
                self.render()

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
    def step(self, action, graspnetwork = False, testfun = False): #TODO: fix this so that we can rotate the hand
        total_reward = 0
        self._get_trans_mat_wrist_pose()
        if len(action)==4:
            action=[0,0,action[0],action[1],action[2],action[3]]

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
                    if (self.orientation == 'rotated') & (action[2]<=0):
                        slide_vector=[-slide_vector[0],-slide_vector[1],slide_vector[2]]
                    else:
                        slide_vector=[-slide_vector[0],-slide_vector[1],slide_vector[2]]
                else:

                    if (self.orientation == 'rotated')&(action[2]<=0):
                        slide_vector=[-slide_vector[0],-slide_vector[1],slide_vector[2]]
                    else:
                        slide_vector=[-action[0],-action[1],action[2]]#np.matmul(self.Twf[0:3,0:3],action[0:3])

                for i in range(3):
                    self._sim.data.ctrl[(i)*2] = slide_vector[i]
                    if self.step_coords=='rotated':
                        self._sim.data.ctrl[i+6] = action[i+3]+0.05
                    else:
                        self._sim.data.ctrl[i+6] = action[i+3]
                    self._sim.data.ctrl[i*2+1]=stuff[i]
                self._sim.step()
            #print('slide vector',slide_vector)
        else:
            for _ in range(self.frame_skip):
                joint_velocities = action[0:7]
                finger_velocities=action[7:]

                for i in range(len(joint_velocities)):

                    self._sim.data.ctrl[i+10] = joint_velocities[i]
                for i in range(len(finger_velocities)):
                    self._sim.data.ctrl[i+7] = finger_velocities[i]
                self._sim.step()
        obs = self._get_obs(test=True)

        if not graspnetwork:
            if not testfun:
                ### Get this reward for RL training ###
                total_reward, info, done = self._get_reward()
            else:
                total_reward, info, done = self._get_reward(test = True)
        else:
            ### Get this reward for grasp classifier collection ###
            total_reward, info, done = self._get_reward_DataCollection()

        return obs, total_reward, done, info

    def add_site(self,world_site_coords,keep_sites=False):
        if not(keep_sites):
            self.site_count=0
        xml_file=open(self.file_dir+self.filename,"r")
        xml_contents=xml_file.read()
        xml_file.close()
        starting_point=xml_contents.find('<body name="root" pos="0 0 0">')
        site_point=xml_contents.find('\n',starting_point)
        end_site_point=xml_contents.find('            <camera name="camera"')
        site_text=f'    		<site name="site{self.site_count}" type="cylinder" size="0.001 0.2" rgba="25 0.5 0.7 1" pos="{world_site_coords[0]} {world_site_coords[1]} {world_site_coords[2]}" euler="0 0 0"/>\n'
        self.site_count+=1
        second_site_text=f'    		<site name="site{self.site_count}" type="cylinder" size="0.001 0.2" rgba="25 0.5 0.7 1" pos="{world_site_coords[0]} {world_site_coords[1]} {world_site_coords[2]}" euler="0 1.5707963267948966 0"/>\n'
        self.site_count+=1
        new_thing=xml_contents[0:site_point+1]+site_text+second_site_text
        if keep_sites:
            new_thing=new_thing+xml_contents[site_point+1:]
        else:
            new_thing=new_thing+xml_contents[end_site_point:]
        
        xml_file=open(self.file_dir+self.filename,"w")
        xml_file.write(new_thing)
        xml_file.close()
        
        self._model = load_model_from_path(self.file_dir + self.filename)
        self._sim = MjSim(self._model) 
        object_location=self._get_obj_size()
        states=[self._sim.data.qpos[0],self._sim.data.qpos[1],self._sim.data.qpos[2],self._sim.data.qpos[3],self._sim.data.qpos[5],self._sim.data.qpos[7],object_location[0],object_location[1],object_location[2]]
        self._set_state(np.array(states))
        self._get_trans_mat_wrist_pose()
        
    def test_self(self):
        shapes=['Cube','Cylinder','Cone1','Cone2','Bowl','Rbowl','Bottle','TBottle','Hour','Vase','Lemon']
        sizes=['S','M','B']
        keys=["CubeS","CubeB","CylinderS","CylinderB","Cone1S","Cone1B","Cone2S","Cone2B","Vase1S","Vase1B","Vase2S","Vase2B"]
        key=random.choice(keys)
        self.reset(obj_params=[key[0:-1],key[-1]])
        print('testing shape',key)
        self._get_obs(test = True)
        print('testing step in global coords')
        action=[0,0,0,0]
        self.step_coords='global'
        start_obs=self._get_obs(state_rep='global')
        for i in range(150):
            action[0]=np.random.rand()-0.2
            self.step(action)
        end_obs=self._get_obs(state_rep='global')    
        if (abs(start_obs[18]-end_obs[18])>0.001)|(abs(start_obs[19]-end_obs[19])>0.001):
            print('test failed. x/y position changed when it should not have, check step function')
        else:
            print('test passed')
        print('printing test step in local coords')
        self.reset(obj_params=[key[0:-1],key[-1]])
        self.step_coords='local'
        start_obs=self._get_obs()
        for i in range(150):
            action[0]=np.random.rand()-0.2
            self.step(action)
        end_obs=self._get_obs()
        if (abs(start_obs[18]-end_obs[18])>0.001)|(abs(start_obs[19]-end_obs[19])>0.001):
            print('test failed. x/y position changed when it should not have, check step function')
        else:
            print('test passed')
        print('no current test for 6 axis motion, step tests finished.')
        print('begining shape test')
        bad_shapes=[]
        for shape in shapes:            
            for size in sizes:
                self.reset(obj_params=[shape,size])
                self.render()
                a=input('obj shape and size',shape,size,'. Is this correct y/n?')
                if a.lower()=='y':
                    print('shape passed')
                else:
                    print('shape failed. recording')
                    bad_shapes.append([shape,size])
        if bad_shapes==[]:
            print('all shapes and sizes are accurate, tests finished')
        else:
            print('the following are shapes that were not correct. Look at the xml files.')
            print(bad_shapes)       
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
    
