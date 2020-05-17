#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Kinova 3-fingered gripper in mujoco environment
# Summer 2019

###############


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
from classifier_network import LinearNetwork
import re
from scipy.stats import triang
# resolve cv2 issue 
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# frame skip = 20
# action update time = 0.002 * 20 = 0.04
# total run time = 40 (n_steps) * 0.04 (action update time) = 1.6

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
            self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1.xml"
            #self._model,self.obj_size,self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"),'s',"/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
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
        # self.viewer = None
        self.contacts=self._sim.data.ncon   # The number of contacts in the simulation environment
        self.Tfw=np.zeros([4,4])   # The trasfer matrix that gets us from the world frame to the local frame
        self.wrist_pose=np.zeros(3)  # The wrist position in world coordinates
        #print(self.contacts)
        ##### Indicate object size (Nigel, data collection only) ###### 
        #self.obj_size = "m"
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
        # self.action_space = spaces.Box(low=np.array([-0.3, -0.3, -0.3, -0.3]), high=np.array([0.3, 0.3, 0.3, 0.3]), dtype=np.float32) # Velocity action space
        # self.action_space = spaces.Box(low=np.array([-1.5, -1.5, -1.5, -1.5]), high=np.array([1.5, 1.5, 1.5, 1.5]), dtype=np.float32) # Position action space
        
        
        self.const_T=np.array([[0,-1,0,0],[0,0,1,0],[-1,0,0,0],[0,0,0,1]])  #Transfer matrix from world frame to un-modified hand frame

        self.frame_skip = frame_skip # Used in step. Number of frames you go through before you reach the next step
        self.all_states = None  # This is the varriable we use to save the states before they are sent to the simulator when we are resetting.

        self.state_rep = "local" # change accordingly
        # self.action_space = spaces.Box(low=np.array([-0.2]), high=np.array([0.2]), dtype=np.float32)
        # self.action_space = spaces.Box(low=np.array([-0.8, -0.8, -0.8]), high=np.array([0.8, 0.8, 0.8]), dtype=np.float32)



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
        trained_model = "/home/orochi/NCSGen-updated/NCSGen-master/gym-kinova-gripper/trained_model_01_23_20_0111.pt"
        #trained_model = "/home/orochi/NCSGen-updated/NCSGen-master/gym-kinova-gripper/trained_model_01_23_20_2052local.pt"
        # self.Grasp_net = GraspValid_net(54).to(device) 
        # trained_model = "/home/graspinglab/NCS_data/ExpertTrainedNet_01_04_20_0250.pt"
        #print(trained_model)
        model = torch.load(trained_model)
        self.Grasp_net.load_state_dict(model)
        self.Grasp_net.eval()
        

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

    # get 3D transformation matrix of each joint
    def _get_trans_mat_wrist_pose(self):
        #print(self.wrist_pose)
        self.wrist_pose=np.copy(self._sim.data.get_geom_xpos('palm'))
        #print(self.wrist_pose)
        Rfa=np.transpose(self._sim.data.get_geom_xmat('palm'))
        Tfa=np.zeros([4,4])
        Twf=np.zeros([4,4])
        Tfa[0:3,0:3]=Rfa
        Tfa[3,3]=1
        Tfw=np.matmul(Tfa,self.const_T)
        self.wrist_pose=self.wrist_pose+np.matmul(np.transpose(Tfw[0:3,0:3]),[0.0,0.06,0.0])
        Tfw[0:3,3]=np.matmul(-np.transpose(Tfw[0:3,0:3]),self.wrist_pose)
        self.Tfw=Tfw 
        Twf[0:3,0:3]=np.transpose(Tfw[0:3,0:3])
        Twf[3,3]=1
        Twf[0:3,3]=-np.matmul(Twf[0:3,0:3],Tfw[0:3,3])
        self.Twf=Twf

    # This thing is brooooken
    def _get_local_pose(self, mat):
        rot_mat = []
        trans = []
        # print(mat)
        for i in range(3):
            orient_temp = []

            for j in range(4):
                if j != 3:
                    orient_temp.append(mat[i][j])
                elif j == 3:
                    trans.append(mat[i][j])
            rot_mat.append(orient_temp)
        pose = list(trans) # + list(euler_vec)

        return pose

    def _get_joint_states(self):
        arr = []
        for i in range(7):
            arr.append(self._sim.data.sensordata[i])

        return arr # it is a list

    # return global or local transformation matrix
    def _get_obs(self, state_rep=None):  #TODO: Add or subtract elements of this to match the discussions with Ravi and Cindy
        if state_rep == None:
            state_rep = self.state_rep

        # states rep
        obj_pose = self._get_obj_pose()
        obj_pose = np.copy(obj_pose)
        #print('obj pose at start of get_obs,',obj_pose)
        #sim_wrist_pose  = self._sim.data.get_geom_xpos("palm")
        #wrist_pos1=self._sim.data.get_geom_xpos("palm")
        #print('wrist pose before obs,', wrist_pos1)
        #self.wrist_pose=np.copy(sim_wrist_pose)
        #print(wrist_pose[1])
        self._get_trans_mat_wrist_pose()
        #print('wrist_pose', wrist_pose)
        x_angle,z_angle = self._get_angles(self.wrist_pose)
        #print(x_angle,z_angle)
        joint_states = self._get_joint_states()
        obj_size = self._sim.model.geom_size[-1] 
        finger_obj_dist = self._get_finger_obj_dist()
        range_data=self._get_rangefinder_data()
        #palm = self._get_trans_mat(["palm"])[0]
        finger_joints = ["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
        
        fingers_6D_pose = []
        #print('palm xmat times tfw', np.matmul(self._sim.data.get_geom_xmat('palm'),np.array([[-1,0,0],[0,-1,0],[0,0,1]])))

        #Tfw[2,3]=-wrist_pose[-1]
        
        #print('Translation matrix Tfw:',Tfw)
        #print('global object location', obj_pose)
        #print('global wrist location', wrist_pose)
        if state_rep == "global":            
            for joint in finger_joints:
                # rot_mat = R.from_dcm(self._sim.data.get_geom_xmat(joint))
                # euler_vec = rot_mat.as_euler('zyx', degrees=True)
                trans = self._sim.data.get_geom_xpos(joint)
                trans = list(trans)
                # trans += list(euler_vec)
                for i in range(3):
                    fingers_6D_pose.append(trans[i])
            fingers_dot_prod = self._get_fingers_dot_product(fingers_6D_pose)
#            print('start:',np.shape(fingers_6D_pose))
#            print(np.shape(wrist_pose))
#            print(np.shape(obj_pose))
#            print(np.shape(joint_states))
#            print('(3,0)')
#            print(np.shape(finger_obj_dist))
#            print('(1,0)')
#            print(np.shape(fingers_dot_prod))
            fingers_6D_pose = fingers_6D_pose + list(self.wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [x_angle, z_angle] + fingers_dot_prod + range_data #+ [self.obj_shape]
            #print(wrist_pose)
            #print(fingers_6D_pose[21])
            #print(obj_pose)
            # pdb.set_trace()

        elif state_rep == "local":
            
            
            for joint in finger_joints:
                # rot_mat = R.from_dcm(self._sim.data.get_geom_xmat(joint))
                # euler_vec = rot_mat.as_euler('zyx', degrees=True)
                trans = self._sim.data.get_geom_xpos(joint)
                trans_for_roation=np.append(trans,1)
                trans_for_roation=np.matmul(self.Tfw,trans_for_roation)
                trans = trans_for_roation[0:3]
                trans = list(trans)
                # trans += list(euler_vec)
                for i in range(3):
                    fingers_6D_pose.append(trans[i])
            fingers_dot_prod = self._get_fingers_dot_product(fingers_6D_pose)
            wrist_for_roation=np.append(self.wrist_pose,1)
            wrist_for_roation=np.matmul(self.Tfw,wrist_for_roation)
            wrist_pose = wrist_for_roation[0:3]
            obj_for_roation=np.append(obj_pose,1)
            obj_for_roation=np.matmul(self.Tfw,obj_for_roation)
            obj_pose = obj_for_roation[0:3]
#            print('start:',np.shape(fingers_6D_pose))
#            print(np.shape(wrist_pose))
#            print(np.shape(obj_pose))
#            print(np.shape(joint_states))
#            print('(3,0)')
#            print(np.shape(finger_obj_dist))
#            print('(1,0)')
#            print(np.shape(fingers_dot_prod))
            #print('local object location', obj_pose)
            #print('local wrist location', wrist_pose)
            #print('local finger location', fingers_6D_pose)
            fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [x_angle, z_angle] + fingers_dot_prod + range_data #+ [self.obj_shape]
            #print(wrist_pose)
            #print(fingers_6D_pose[21])
            #print(obj_pose)
            # pdb.set_trace()
            
            '''
            finger_joints_local = []
            palm_inverse = np.linalg.inv(palm)
            for joint in range(len(finger_joints_transmat)):
                joint_in_local_frame = np.matmul(finger_joints_transmat[joint], palm_inverse)
                pose = self._get_local_pose(joint_in_local_frame)
                for i in range(3):
                    fingers_6D_pose.append(pose[i])
            fingers_dot_prod = self._get_fingers_dot_product(fingers_6D_pose)
#            print('start:',np.shape(fingers_6D_pose))
#            print(np.shape(wrist_pose))
#            print(np.shape(obj_pose))
#            print(np.shape(joint_states))
#            print('(3,0)')
#            print(np.shape(finger_obj_dist))
#            print('(1,0)')
#            print(np.shape(fingers_dot_prod))
            fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [obj_dot_prod] + fingers_dot_prod + range_data
            print('object pose',obj_pose)
            print('wrist pose', wrist_pose)
            '''
        elif state_rep == "metric":
            fingers_6D_pose = self._get_rangefinder_data()
            fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [x_angle, z_angle] #+ fingers_dot_prod

        elif state_rep == "joint_states":
            fingers_6D_pose = joint_states + list(obj_pose) + [obj_size[0], obj_size[1], obj_size[2]*2] + [x_angle, z_angle] #+ fingers_dot_prod

        #print('obj pose at end of get_obs,',self._get_obj_pose())
        #wrist_pos1=self._sim.data.get_geom_xpos("palm")
        #print('wrist pose after obs,', wrist_pos1)
        # print(joint_states[0:4])
        return fingers_6D_pose 

    def _get_finger_obj_dist(self):
        # finger_joints = ["palm", "f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
        finger_joints = ["palm_1", "f1_prox","f1_prox_1", "f2_prox", "f2_prox_1", "f3_prox", "f3_prox_1", "f1_dist", "f1_dist_1", "f2_dist", "f2_dist_1", "f3_dist", "f3_dist_1"]

        obj = self._get_obj_pose()
        dists = []
        for i in finger_joints:
            pos = self._sim.data.get_site_xpos(i)
            dist = np.absolute(pos[0:2] - obj[0:2])
            dist[0] -= 0.0175
            temp = np.linalg.norm(dist)
            dists.append(temp)
            # pdb.set_trace()
        return dists


    # get range data from 1 step of time 
    # Uncertainty: rangefinder could only detect distance to the nearest geom, therefore it could detect geom that is not object
    def _get_rangefinder_data(self):
        range_data = []
        for i in range(17):
            if self._sim.data.sensordata[i+7]==-1:
                a=6
            else:
                a=self._sim.data.sensordata[i+7]
            range_data.append(a)

        return range_data

    def _get_obj_pose(self):
        arr = self._sim.data.get_geom_xpos("object")
        return arr

    def _get_fingers_dot_product(self, fingers_6D_pose):  #TODO: Make it so that this returns the angles for the finger joints or don't call it
        fingers_dot_product = []
        for i in range(6):
            fingers_dot_product.append(self._get_angles(fingers_6D_pose[3*i:3*i+3]))
        return fingers_dot_product

    # Function to return dot product based on object location
    def _get_angles(self, obj_state): #TODO if we are using get_fingers_dot_product (Shitty name btw), change so that this uses the object state as the location of the vertex of the angle
        # obj_state = self._get_obj_pose()
        obj_pose = self._get_obj_pose()
        #print(obj_pose)
        #obj_pose=np.copy(obj_pose)
        self._get_trans_mat_wrist_pose()
        local_obj_pos=np.copy(obj_pose)
        local_obj_pos=np.append(local_obj_pos,1)
        #print(self.Tfw)
        local_obj_pos=np.matmul(self.Tfw,local_obj_pos)
        #wrist_pose  = self._sim.data.get_geom_xpos("palm")
        #obj_rot = self._sim.data.get_body_xquat("j2s7s300_link_7")
        #print("object quaternian," ,obj_rot)
        #print(wrist_pose[1])
        #wrist_pose=np.copy(wrist_pose)
        #wrist_pose[1]=wrist_pose[1]-0.06
        #print('object position',local_obj_pos)
        obj_wrist = local_obj_pos[0:3]/np.linalg.norm(local_obj_pos[0:3])
        #print(obj_wrist)
        center_line = np.array([0,1,0])
        z_dot = np.dot(obj_wrist[0:2],center_line[0:2])
        z_angle = np.arccos(z_dot/np.linalg.norm(obj_wrist[0:2]))
        x_dot = np.dot(obj_wrist[1:3],center_line[1:3])
        x_angle = np.arccos(x_dot/np.linalg.norm(obj_wrist[1:3]))
        #print('da two angles,',x_angle,z_angle)
        
        
        return x_angle,z_angle

    '''
    Testing reward function (not useful, saved it for testing purposes)
    '''
    def _get_dist_reward(self, state, action):
        
        f1 = self._sim.data.get_geom_xpos("f1_dist")

        target_pos = np.array([0.05813983, 0.01458329])
        target_vel = np.array([0.0])

        f1_pos_err = (target_pos[0] - state[0])**2 + (target_pos[1] - state[1])**2
        # f1_curr = math.sqrt((0.05813983 - state[0])**2 + (0.01458329 - state[1])**2)
        # f1_curr = (0.058 - state[0])**2 + (0.014 - state[1])**2
        f1_vel_err = (target_vel[0] - action)**2

        f1_pos_reward = 12*(math.exp(-100*f1_pos_err) - 1)
        f1_vel_reward = 4.5*(math.exp(-f1_vel_err) - 1)

        reward = f1_pos_reward 

        return reward

    # def _expertvelocity_reward(self, action):
        # input 

    def _get_reward_DataCollection(self):
        obj_target = 0.2
        obs = self._get_obs(state_rep="global") 
        #print(obs[23])
        if abs(obs[23] - obj_target) < 0.005 or (obs[23] >= obj_target):
            lift_reward = 1
            done = True
            #print('condition 1: ',abs(obs[23] - obj_target))
            #print('condition 2: ',obs[23])
        elif obs[5]>obj_target+0.05:
            lift_reward=0.0
            done=True
        else:
            lift_reward = 0
            done = False        
        #print('reward', done)
        return lift_reward, {}, done

    '''
    Reward function (Actual)
    '''
    def _get_reward(self):

        # object height target
        obj_target = 0.2

        # Grasp reward
        grasp_reward = 0.0
        obs = self._get_obs(state_rep="global") 
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
             #grasp_reward = outputs
        
        if abs(obs[23] - obj_target) < 0.005 or (obs[23] >= obj_target):
            lift_reward = 50.0
            done = True
            #print('success,', self.obj_shape)
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
        #obj_pose=self._get_obj_pose()
        #print('obj pose after set state,', obj_pose)
        #wrist_pos=self._sim.data.get_geom_xpos("palm")
        #print('wrist pose after set state,', wrist_pos)

    def _get_obj_size(self):
        return self._sim.model.geom_size[-1]


    def set_obj_size(self, default = False):
        hand_param = {}
        hand_param["span"] = 0.15
        hand_param["depth"] = 0.08
        hand_param["height"] = 0.15 # including distance between table and hand

        geom_types = ["box", "cylinder"]#, "sphere"]
        geom_sizes = ["s", "m", "b"]

        geom_type = random.choice(geom_types)
        geom_size = random.choice(geom_sizes)

        # Cube w: 0.1, 0.2, 0.3
        # Cylinder w: 0.1, 0.2, 0.3
        # Sphere w: 0.1, 0.2, 0.3

        # Cube & Cylinder
        width_max = hand_param["span"] * 0.3333 # 5 cm
        width_mid = hand_param["span"] * 0.2833 # 4.25 cm
        width_min = hand_param["span"] * 0.2333 # 3.5 cm
        width_choice = np.array([width_min, width_mid, width_max])

        height_max = hand_param["height"] * 0.80 # 0.12
        height_mid = hand_param["height"] * 0.73333 # 0.11
        height_min = hand_param["height"] * 0.66667 # 0.10
        height_choice = np.array([height_min, height_mid, height_max])

        # Sphere
        # radius_max = hand_param["span"] * 0.
        # radius_mid = hand_param["span"] * 0.2833 
        # radius_min = hand_param["span"] * 0.2333
        # radius_choice = np.array([radius_min, radius_mid, radius_max])

        if default:
            # print("here")
            return "box", np.array([width_choice[1]/2.0, width_choice[1]/2.0, height_choice[1]/2.0])
        else:

            if geom_type == "box": #or geom_type == "cylinder":
                if geom_size == "s":
                    geom_dim = np.array([width_choice[0] / 2.0, width_choice[0] / 2.0, height_choice[0] / 2.0])
                if geom_size == "m":
                    geom_dim = np.array([width_choice[1] / 2.0, width_choice[1] / 2.0, height_choice[1] / 2.0])
                if geom_size == "b":
                    geom_dim = np.array([width_choice[2] / 2.0, width_choice[2] / 2.0, height_choice[2] / 2.0])
            if geom_type == "cylinder":
                if geom_size == "s":
                    geom_dim = np.array([width_choice[0] / 2.0, height_choice[0] / 2.0])
                if geom_size == "m":
                    geom_dim = np.array([width_choice[1] / 2.0, height_choice[1] / 2.0])
                if geom_size == "b":
                    geom_dim = np.array([width_choice[2] / 2.0, height_choice[2] / 2.0])
            # if geom_type == "sphere":
            #     if geom_size == "s":
            #         geom_dim = np.array([radius_choice[0]])
            #     if geom_size == "m":
            #         geom_dim = np.array([radius_choice[1]])
            #     if geom_size == "b":
            #         geom_dim = np.array([radius_choice[2]])

            return geom_type, geom_dim, geom_size
                            
    def gen_new_obj(self, default = False):
        file_dir = "./gym_kinova_gripper/envs/kinova_description"
        filename = "/objects.xml"
        tree = ET.parse(file_dir + filename)
        root = tree.getroot()
        d = default
        next_root = root.find("body")
        # print(next_root)
        # pick a shape and size
        geom_type, geom_dim, geom_size = self.set_obj_size(default = d)
        # if geom_type == "sphere":
        #     next_root.find("geom").attrib["size"] = "{}".format(geom_dim[0])
        if geom_type == "box":
            next_root.find("geom").attrib["size"] = "{} {} {}".format(geom_dim[0], geom_dim[1], geom_dim[2])
        if geom_type == "cylinder":
            next_root.find("geom").attrib["size"] = "{} {}".format(geom_dim[0], geom_dim[1])
            
        next_root.find("geom").attrib["type"] = geom_type
        tree.write(file_dir + "/objects.xml")

        return geom_type, geom_dim, geom_size


    def randomize_initial_pose(self, collect_data, size, shape):
        # geom_type, geom_dim, geom_size = self.gen_new_obj()
        # geom_size = "s"
        geom_size = size
        #self.obj_shape
        # self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml")
        # self._sim = MjSim(self._model)

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
            #x = [0.04, 0.03, -0.04, -0.03]
            #y = [0.0, 0.02, 0.03]
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
        # return rand_x, rand_y, geom_dim[-1]
        #print(rand_x, rand_y)
        return rand_x, rand_y


        # medium x = [0.04, 0.03, 0.02]
        # med y = [0.0, 0.02, 0.03]
        # large x = [0.03, 0.02] 
        # large y = [0.0, 0.02]

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

    def randomize_all(self):
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

    def randomize_initial_pos_data_collection(self):
        # print(self._sim.model.geom_size[-1])
        
        size=self._get_obj_size()
        #print(size)
        
        
        
        #The old way to generate random poses
        '''
        if self.obj_size == "s":
            x = [0.05, 0.04, 0.03, 0.02, -0.05, -0.04, -0.03, -0.02]
            y = [0.0, 0.02, 0.03, 0.04]
            rand_x = random.choice(x)
            if rand_x == 0.05 or rand_x == -0.05:
                rand_y = 0.0
            elif rand_x == 0.04 or rand_x == -0.04:
                rand_y = random.uniform(0.0, 0.02)
            elif rand_x == 0.03 or rand_x == -0.03:
                rand_y = random.uniform(0.0, 0.03)
            elif rand_x == 0.02 or rand_x == -0.02:
                rand_y = random.uniform(0.0, 0.04)
        if self.obj_size == "m":
            x = [0.04, 0.03, 0.02, -0.04, -0.03, -0.02]
            y = [0.0, 0.02, 0.03]
            rand_x = random.choice(x)
            if rand_x == 0.04 or rand_x == -0.04:
                rand_y = 0.0
            elif rand_x == 0.03 or rand_x == -0.03:
                rand_y = random.uniform(0.0, 0.02)
            elif rand_x == 0.02 or rand_x == -0.02:
                rand_y = random.uniform(0.0, 0.03)
        if self.obj_size == "b":
            x = [0.03, 0.02, -0.03, -0.02]
            y = [0.0, 0.02]
            rand_x = random.choice(x)
            if rand_x == 0.03 or rand_x == -0.03:
                rand_y = 0.0
            elif rand_x == 0.02 or rand_x == -0.02:
                rand_y = random.uniform(0.0, 0.02)
        '''
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
        #print(z)
        #print(rand_x, rand_y, z)
        return rand_x, rand_y, z    

    def reset(self,start_pos=None,obj_params=None):
        # x, y = self.randomize_initial_pose(False, "s") # for RL training
        #x, y = self.randomize_initial_pose(True) # for data collection
        objects = self.experiment(1, 2)
        
        shapes=list(objects.keys())
        #print(shapes[0])
        self._get_jacobian()
        '''
        hand_rotation=np.random.normal(0,0.125,3)
        '''
        
        #hand_rotation=np.array([np.pi/2, 0, np.pi/2])
        
        if obj_params !=None:
            if obj_params[0] == "Box":
                if obj_params[1] == "B":
                    #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
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
        '''
        if self.filename=="/kinova_description/j2s7s300_end_effector.xml":
            new_rotation=np.array([0,0,0])+hand_rotation
        else:
            new_rotation=np.array([-1.57,0,-1.57])+hand_rotation
        a,b,c=-hand_rotation[0],-hand_rotation[1],-hand_rotation[2]
        
        #new_motion=np.array([[np.cos(c)*np.cos(b), np.cos(c)*np.sin(b)*np.sin(a)-np.sin(c)*np.cos(a), np.cos(a)*np.sin(b)*np.cos(c)+np.sin(a)*np.sin(c)], \
        #                     [np.sin(c)*np.cos(b), np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c), np.sin(c)*np.sin(b)*np.cos(a)-np.cos(c)*np.sin(a)], \
        #                     [-np.sin(b), np.cos(b)*np.sin(a), np.cos(b)*np.cos(a)]])
        Rx=np.array([[1,0,0],[0, np.cos(a), -np.sin(a)],[0, np.sin(a), np.cos(a)]])
        Ry=np.array([[np.cos(b), 0, np.sin(b)],[0, 1, 0],[-np.sin(b), 0, np.cos(b)]])
        Rz=np.array([[np.cos(c), -np.sin(c), 0],[np.sin(c), np.cos(c), 0],[0, 0, 1]])
        new_motion=np.matmul(Rx,Ry)
        new_motion=np.matmul(new_motion,Rz)
        new_motion=np.matmul(new_motion,np.array([1,0,0]))
        for i in range(3):
            if abs(new_motion[i]) <0.001:
                new_motion[i]=0
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
        #print('THIS IS THE POINT',euler_point)
        contents=re.search("[^\s]+\s[^\s]+\s[^>]+",xml_contents[axis_point:])
        c2_start=contents.start()
        c2_end=contents.end()
        p1=str(new_rotation[0])
        p2=str(new_rotation[1])
        p3=str(new_rotation[2])
        p4=str(new_motion[0])
        p5=str(new_motion[1])
        p6=str(new_motion[2])
        xml_contents=xml_contents[:euler_point+c_start+7] + p1[0:min(5,len(p1))]+ " "+p2[0:min(5,len(p2))] +" "+ p3[0:min(5,len(p3))] \
        + xml_contents[euler_point+c_end-1:axis_point+c2_start+6] + p4[0:min(5,len(p4))]+ " "+p5[0:min(5,len(p5))] +" "+ p6[0:min(5,len(p6))] \
        + xml_contents[axis_point+c2_end-3:]
        #print(xml_contents)
        xml_file=open(self.file_dir+self.filename,"w")
        xml_file.write(xml_contents)
        xml_file.close()
        '''
        
        self._model = load_model_from_path(self.file_dir + self.filename)
        #print()


        
        
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
        
        # self.all_states_1 = np.array([0.0, 0.0, 0.0, 0.0, x, y, 0.05])
        #x,y,z=0,0,0.05
        self.all_states_1 = np.array([0.0, 0.0, 0.0, 0.0, x, y, z])
        #print(x,y,z)
        
        #self.all_states_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0, z])
        self.Grasp_Reward=False
        self.all_states_2 = np.array([0.0, 0.9, 0.9, 0.9, 0.0, -0.01, 0.05])
        self.all_states = [self.all_states_1 , self.all_states_2] 
        random_start = np.random.randint(2)

        self.obj_original_state = np.array([0.05, 0.0])
        self._set_state(self.all_states[0])
        # self.init_dotprod = self._get_angles()
        # self.init_pose = np.array([x, y, 0.05])

        states = self._get_obs()
        obj_pose=self._get_obj_pose()
        deltas=[x-obj_pose[0],y-obj_pose[1],z-obj_pose[2]]
        #print('states',states[21:24],)
        #print(deltas,'deltas')
        #states[21]=x
        #states[22]=y
        #states[23]=z
        
        
        if np.linalg.norm(deltas)>0.05:
            self.all_states_1=np.array([0.0, 0.0, 0.0, 0.0, x+deltas[0], y+deltas[1], z+deltas[2]])
            self.all_states=[self.all_states_1,self.all_states_2]
            self._set_state(self.all_states[0])
            states = self._get_obs()
        
        #print('deltas,',deltas)
        # self.prev_fr = 0.0
        # self.prev_r = 0.0
        self.t_vel = 0
        self.prev_obs = []
        
        return states

    def render(self, mode='human'):
        a=False
        if self._viewer is None:
            self._viewer = MjViewer(self._sim)
            a=True
        self._viewer.render()
        if a:
            self._viewer._paused=True
        
    def close(self):
        if self._viewer is not None:
            self._viewer = None
    
    def pause(self):
        self._viewer._paused=True
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ###################################################
    ##### ---- Action space : Joint Velocity ---- #####
    ###################################################
    def step(self, action): #TODO: fix this so that it doesn't jerk around when sliding and so that we can rotate the hand
        total_reward = 0
        
        #if len(action)==4:
        #    action=[action[0],0,0,0,0,0,action[1],action[2],action[3]]
        hand_velocity=self._sim.data.get_body_xvelp('j2s7s300_link_7')
        hand_pos=self._sim.data.get_geom_xpos("palm")
        #print('hand velocity', hand_velocity)
        #print('had pos', hand_pos)
        self._get_trans_mat_wrist_pose()
        
        
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
        
        
        for _ in range(self.frame_skip):
            slide_vector=np.array([action[0],action[1],action[2]])
            #print('pre',slide_vector)
            slide_vector=np.matmul(self.Tfw[0:3,0:3],slide_vector)
            #print('post',slide_vector)
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
            #self._sim.data.ctrl[-1]=5*hand_pos[-1]
            self._sim.step()
        #print('and we rolled all day', self._sim.data.ctrl)
        obs = self._get_obs()

        ### Get this reward for RL training ###
        #total_reward, info, done = self._get_reward()

        ### Get this reward for data collection ###
        total_reward, info, done = self._get_reward_DataCollection()
        #print('step', done)
        # print(obs[15:18], self._get_angles(obs[15:18]))

        # print(self._get_angles)
        return obs, total_reward, done, info
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
