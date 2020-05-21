#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: Kinova 3-fingered gripper in mujoco environment
# Summer 2019

###############


import gym
from gym import utils, spaces
from gym.utils import seeding
# from gym.envs.mujoco import mujoco_env
import numpy as np
from mujoco_py import MjViewer, load_model_from_path, MjSim
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
import copy
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
            #self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector.xml")
            self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_scyl.xml")
            #self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mbox.xml")
            #self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml")
            #self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bbox.xml")
            #self._model = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml")

            # full_path = file_dir + "/kinova_description/j2s7s300_end_effector_v1.xml"
        else:
            print("CHOOSE EITHER HAND OR ARM")
            raise ValueError
        
        self._sim = MjSim(self._model)
        
        self._viewer = None
        # self.viewer = None

        ##### Indicate object size (Nigel, data collection only) ###### 
        self.obj_size = "b"

        self._timestep = self._sim.model.opt.timestep
        self._torque = [0,0,0,0]
        self._velocity = [0,0,0,0]

        self._jointAngle = [0,0,0,0]
        self._positions = [] # ??
        self._numSteps = 0
        self._simulator = "Mujoco"
        self.action_scale = 0.0333
        self.max_episode_steps = 150
        # Parameters for cost function
        self.state_des = 0.20 
        self.initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.frame_skip = frame_skip
        self.all_states = None
        self.action_space = spaces.Box(low=np.array([-0.8, -0.8, -0.8, -0.8]), high=np.array([0.8, 0.8, 0.8, 0.8]), dtype=np.float32) # Velocity action space
        # self.action_space = spaces.Box(low=np.array([-0.3, -0.3, -0.3, -0.3]), high=np.array([0.3, 0.3, 0.3, 0.3]), dtype=np.float32) # Velocity action space
        # self.action_space = spaces.Box(low=np.array([-1.5, -1.5, -1.5, -1.5]), high=np.array([1.5, 1.5, 1.5, 1.5]), dtype=np.float32) # Position action space
        self.state_rep = "global" # change accordingly
        # self.action_space = spaces.Box(low=np.array([-0.2]), high=np.array([0.2]), dtype=np.float32)
        # self.action_space = spaces.Box(low=np.array([-0.8, -0.8, -0.8]), high=np.array([0.8, 0.8, 0.8]), dtype=np.float32)

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
        '''
        self.Grasp_net = GraspValid_net(48).to(device) 
        trained_model = "/home/graspinglab/NCS_data/data_cube_9_grasp_classifier_10_17_19_1734.pt"

        # self.Grasp_net = GraspValid_net(54).to(device) 
        # trained_model = "/home/graspinglab/NCS_data/ExpertTrainedNet_01_04_20_0250.pt"
        model = torch.load(trained_model)
        self.Grasp_net.load_state_dict(model)
        self.Grasp_net.eval()
        '''

    # get 3D transformation matrix of each joint
    def _get_trans_mat(self, joint_geom_name):
        finger_joints = joint_geom_name    
        finger_pose = []
        empty = np.array([0,0,0,1])
        for each_joint in finger_joints:
            arr = []
            for axis in range(3):
                temp = np.append(self._sim.data.get_geom_xmat(each_joint)[axis], self._sim.data.get_geom_xpos(each_joint)[axis])
                arr.append(temp)
            arr.append(empty)
            arr = np.array(arr)
            finger_pose.append(arr)    
        return np.array(finger_pose)


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
    def _get_obs(self, state_rep=None):
        if state_rep == None:
            state_rep = self.state_rep

        # states rep
        obj_pose = self._get_obj_pose()
        obj_dot_prod = self._get_dot_product(obj_pose)
        wrist_pose  = self._sim.data.get_geom_xpos("palm")
        joint_states = self._get_joint_states()
        obj_size = self._sim.model.geom_size[-1] 
        finger_obj_dist = self._get_finger_obj_dist()
        range_data=self._get_rangefinder_data()
        palm = self._get_trans_mat(["palm"])[0]
        finger_joints = ["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"]
        finger_joints_transmat = self._get_trans_mat(["f1_prox", "f2_prox", "f3_prox", "f1_dist", "f2_dist", "f3_dist"])
        fingers_6D_pose = []
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
            fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [obj_dot_prod] + fingers_dot_prod + range_data
            # pdb.set_trace()

        elif state_rep == "local":
            finger_joints_local = []
            palm_inverse = np.linalg.inv(palm)
            for joint in range(len(finger_joints_transmat)):
                joint_in_local_frame = np.matmul(finger_joints_transmat[joint], palm_inverse)
                pose = self._get_local_pose(joint_in_local_frame)
                for i in range(3):
                    fingers_6D_pose.append(pose[i])
            fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [obj_dot_prod] #+ fingers_dot_prod

        elif state_rep == "metric":
            fingers_6D_pose = self._get_rangefinder_data()
            fingers_6D_pose = fingers_6D_pose + list(wrist_pose) + list(obj_pose) + joint_states + [obj_size[0], obj_size[1], obj_size[2]*2] + finger_obj_dist + [obj_dot_prod] #+ fingers_dot_prod

        elif state_rep == "joint_states":
            fingers_6D_pose = joint_states + list(obj_pose) + [obj_size[0], obj_size[1], obj_size[2]*2] + [obj_dot_prod] #+ fingers_dot_prod

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
            range_data.append(self._sim.data.sensordata[i+7])

        return range_data

    def _get_obj_pose(self):
        arr = self._sim.data.get_geom_xpos("object")
        return arr

    def _get_fingers_dot_product(self, fingers_6D_pose):
        fingers_dot_product = []
        for i in range(6):
            fingers_dot_product.append(self._get_dot_product(fingers_6D_pose[3*i:3*i+3]))
        return fingers_dot_product

    # Function to return dot product based on object location
    def _get_dot_product(self, obj_state):
        # obj_state = self._get_obj_pose()
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
        # grasp_reward = 0.0
        obs = self._get_obs(state_rep="global") 
        # inputs = torch.FloatTensor(np.array(obs[0:48])).to(device)
        # if np.max(np.array(obs[41:47])) < 0.035 or np.max(np.array(obs[35:41])) < 0.015: 
        #     outputs = self.Grasp_net(inputs).cpu().data.numpy().flatten()
        #     if outputs == 1.0:
        #         grasp_reward = 5.0
        #     else:
        #         grasp_reward = 5.0
            # grasp_reward = outputs
        
        if abs(obs[23] - obj_target) < 0.005 or (obs[23] >= obj_target):
            lift_reward = 50.0
            done = True
        else:
            lift_reward = 0.0
            done = False

        finger_reward = -np.sum((np.array(obs[41:47])) + (np.array(obs[35:41])))

        reward = 0.2*finger_reward + lift_reward #+ grasp_reward

        return reward, {}, done

    # only set proximal joints, cuz this is an underactuated hand
    def _set_state(self, states):
        self._sim.data.qpos[0] = states[0]
        self._sim.data.qpos[1] = states[1]
        self._sim.data.qpos[3] = states[2]
        self._sim.data.qpos[5] = states[3]
        self._sim.data.set_joint_qpos("object", [states[4], states[5], states[6], 1.0, 0.0, 0.0, 0.0])
        self._sim.forward()

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


    def randomize_initial_pose(self, collect_data, size):
        # geom_type, geom_dim, geom_size = self.gen_new_obj()
        # geom_size = "s"
        geom_size = size

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
            x = [0.04, 0.03, 0.02, -0.04, -0.03, -0.02]
            y = [0.0, 0.02, 0.03]
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
        # print(rand_x, rand_y)
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
                # objects["mbox"] = "/kinova_description/j2s7s300_end_effector_v1_mbox.xml"

            # Exp 1 Stage 2: Change shape
            if stage_num == 2:
                objects["scyl"] = "/kinova_description/j2s7s300_end_effector_v1_scyl.xml"
                objects["bcyl"] = "/kinova_description/j2s7s300_end_effector_v1_bcyl.xml"

                # Testing Exp 1 Stage 2
                # objects["mcyl"] = "/kinova_description/j2s7s300_end_effector_v1_mcyl.xml"
        
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

        objects = self.experiment(1, 1)

        random_shape = np.random.choice(list(objects.keys()))
        self._model = load_model_from_path(self.file_dir + objects[random_shape])
        self._sim = MjSim(self._model)
        # print (random_shape)
        if random_shape == "sbox" or random_shape == "scyl":
            x, y = self.randomize_initial_pose(False, "s")
            z = 0.05
        elif random_shape == "mbox" or random_shape == "mcyl":
            x, y = self.randomize_initial_pose(False, "m")
            z = 0.055
        elif random_shape == "bbox" or random_shape == "bcyl":
            x, y = self.randomize_initial_pose(False, "b")
            z = 0.06            
        else:
            print("size and shape are incorrect")
            raise ValueError

        return x, y, z

    def randomize_initial_pos_data_collection(self):
        # print(self._sim.model.geom_size[-1])
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

        z = self._sim.model.geom_size[-1][-1]
        return rand_x, rand_y, z    

    def reset(self):
        # x, y = self.randomize_initial_pose(False, "s") # for RL training
        # x, y = self.randomize_initial_pose(True) # for data collection
        # x, y, z = self.randomize_all() # for RL training all objects
        # x, y, z = self.randomize_initial_pos_data_collection()
        # self.all_states_1 = np.array([0.0, 0.0, 0.0, 0.0, x, y, 0.05])
        x,y,z=0,0,0
        self.all_states_1 = np.array([0.0, 0.0, 0.0, 0.0, x, y, z])

        self.all_states_2 = np.array([0.0, 0.9, 0.9, 0.9, 0.0, -0.01, 0.05])
        self.all_states = [self.all_states_1 , self.all_states_2] 
        random_start = np.random.randint(2)

        self.obj_original_state = np.array([0.05, 0.0])
        self._set_state(self.all_states[0])
        # self.init_dotprod = self._get_dot_product()
        # self.init_pose = np.array([x, y, 0.05])

        states = self._get_obs()

        # self.prev_fr = 0.0
        # self.prev_r = 0.0
        self.t_vel = 0
        self.prev_obs = []
        return states

    def render(self, mode='human'):
        if self._viewer is None:
            self._viewer = MjViewer(self._sim)
        self._viewer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ###################################################
    ##### ---- Action space : Joint Velocity ---- #####
    ###################################################
    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            if action[0] < 0.0:
                self._sim.data.ctrl[0] = 0.0
            else:    
                self._sim.data.ctrl[0] = (action[0] / 0.8) * 0.2
                # self._sim.data.ctrl[0] = action[0]

            for i in range(3):
                # vel = action[i]
                if action[i+1] < 0.0:
                    self._sim.data.ctrl[i+1] = 0.0
                else:    
                    self._sim.data.ctrl[i+1] = action[i+1]
            self._sim.step()

        obs = self._get_obs()

        ### Get this reward for RL training ###
        #total_reward, info, done = self._get_reward()

        ### Get this reward for data collection ###
        total_reward, info, done = self._get_reward_DataCollection()
        #print('step', done)
        # print(obs[15:18], self._get_dot_product(obs[15:18]))

        # print(self._get_dot_product)
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
