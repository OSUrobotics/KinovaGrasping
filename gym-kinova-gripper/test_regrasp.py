# This script is to teleoperate Kinova gripper in the mujoco env

import gym
import numpy as np
import pdb
from classifier_network import LinearNetwork, ReducedLinearNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import triang
from scipy.spatial.transform import Rotation as R
#import serial
import matplotlib.pyplot as plt
#import optimizer
import csv
import time

def pid(kp, ki, kd, error, ddterror, error_sum):
    #if np.abs(error) <= .5:
    #    return 0
    return (kp*error) + (kd*ddterror) +(ki*error_sum)

#Random Vector generation from:
#https://towardsdatascience.com/the-best-way-to-pick-a-unit-vector-7bd0cc54f9b
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


action = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0, 0])

t = 0
coords='local'

episode_obs=[]
value=0

s_or_f=[]
last_obs=[]
obs=[]
act=np.array([0.3,0.3,0.3,0,0])

#Actions [slide_x, slide_y, slide_z, finger_vel,finger_vel,finger_vel, wrist_vel_y, wrist_vel_z]
actions=[[0,0,0,0.3,0.3,0.3, 0, 0],[0,0,0,0.3,0.3,0,0,0],[0,0,0,0,0,0.3,0,0], [0,0,0,0,0,0,.3,0], [0,0,0,0,0,0,0,.3]]
poses=[[0.0,-0.03],[0.02,-0.03],[-0.02,-0.03],[-0.05,0],[-0.01,-0.035],[0.01,-0.035],[0.05,0],[-0.02,0.03],[0.0,0.03],[0.02,0.03]]

#Grasp Classifier
#grasp_classifier = ReducedLinearNetwork()
#grasp_classifier=grasp_classifier.float()
#grasp_classifier.load_state_dict(torch.load('trained_model_05_30_20_1119local.pt'))
#grasp_classifier=grasp_classifier.float()
#grasp_classifier.eval()
#print('Grasp Classifier Loaded')

#env.env.pid=True
#What is this outer loop?
for f in range(1):#was 3

    #Episodes
    for k in range(1):#was 10
        #Setup env
        #Add random vector
        env.add_vec_site([0,0,0], vec)
        env.reset(shape_keys=["RGmBox"],obj_params=["RGCube","M"],hand_orientation="random")        


        #Default action
        action=np.append([0,0,0],act)
        print('reset')
        error_sum_y = 0
        error_sum_z = 0

        x_angles = []
        y_angles = []
        z_angles= []
        
        #Timesteps
        for i in range(100):

            #want local_stem = [1,0,0]
            local_stem = np.matmul(env.env.Tfw[0:3,0:3], vec)
            #print('local stem dir: ',local_stem)
            
            #stem is X
            #dot product stem w/ [v1,v2,v3]=0 to get Y
            #   Set v1&v2 to random value
            #   solve for v3
            #Normalize new vec Y
            #cross product of X and Y to get Z
            #get rot matrix for world->stem [X,Y,Z]
            
            #Stem
            X = vec

            #Solve for v3
            v0,v1 =.1,.1
            v2 = (-(vec[0]*v0)-(vec[1]*v1))/vec[2]
            Y = np.array([v0,v1,v2]) / np.linalg.norm([v0,v1,v2])
            
            Z = np.cross(X,Y)
            #Stem to world
            Rsw = np.array([X,Y,Z])
            print('RSW:',Rsw)
            print('X:', X)
            #local to stem
            #Rfs = np.matmul(env.env.Tfw[0:3,0:3], Rws)
            #euler_angles = R.from_matrix(Rfs).as_quat()#as_euler('xyz', degrees=False)
            
            #World to stem
            Q_target = R.from_matrix(Rsw)
            #Local to world
            Q_current = R.from_matrix(env.env.Tfw[0:3,0:3]).inv()
            #print('Qt:', Q_target.as_quat())
            #print('Qc:', Q_current.as_quat())
            # calculate the rotation between current and target orientations
            #q_r = transformations.quaternion_multiply(q_target, transformations.quaternion_conjugate(q_e))            
            Q_r = (Q_target*Q_current).as_quat()
            print('Qr:', Q_r)
            xcol = np.array([1,0,0]).reshape(3, 1)
            mat_Q_r=R.from_quat(Q_r).as_matrix() 
            print('Qr[1,0,0]T:\n', mat_Q_r.dot(xcol))
            #convert rotation quaternion to Euler angle forces            
            #u_task[3:] = ko * q_r[1:] * np.sign(q_r[0])

            #print('Qc xyz:', Q_current.as_euler('xyz'))
            local_euler = Q_r[:3] * np.sign(Q_r[3])
            euler_angles = R.from_euler('xyz', local_euler).as_matrix()
            euler_angles = R.from_matrix(np.matmul(euler_angles, env.env.Tfw[0:3,0:3])).as_euler('xyz')
            #print('global: ', euler_angles)
            print('local: ', local_euler)

            x_angles.append(local_euler[0])
            y_angles.append(local_euler[1])
            z_angles.append(local_euler[2])

            if i == 0:
                prev_angles = local_euler.copy()
            error_sum_y += local_euler[1]
            error_sum_z += local_euler[2]

            roty = pid(-.025, 0, 0, local_euler[1], local_euler[1]-prev_angles[1], error_sum_y)
            rotz = pid(-.025, 0, 0, local_euler[2], local_euler[2]-prev_angles[2], error_sum_z)
            prev_angles = local_euler.copy()

            #Set action to lift
            #if i == 150:
            #    print('move in z')
            #    action=np.array([0.15,0.05, 0.05, 0.05])
            #    env.env.pid=True
            #    last_obs.append(obs)
            #testing

            action = np.array([0,0,0,0,0,0,rotz,roty])

            #curr_pitch = env.env._sim.data.get_joint_qpos('j2s7s300_joint_wrist_pitch')
            #curr_yaw = env.env._sim.data.get_joint_qpos('j2s7s300_joint_wrist_yaw')
            #print("curr_pitch: ", curr_pitch)
            #print("curr_yaw: ", curr_yaw)

            #print("diff pitch:", curr_pitch - Q_current.as_euler('xyz')[1])
            #action = np.array([0,0,0,0,0,0,local_euler[2]+curr_yaw,local_euler[1]+curr_pitch])                

            #if i == 25:
            #    action = np.array([0,0,0,0,0,0,0,0.3])

            #????
            #if coords=='global':
            #    temp=np.array([action[0],action[1],action[2],1])
            #    action[0:3]=np.matmul(env.Twf[0:3,0:3],action[0:3])
            
            #print("Action: ", action)
            #STEP
            obs, reward, done, _ = env.step(action)
            env.render()
            
            network_feed=obs[21:24]
            #print('local obs',obs[21:24])
            network_feed=np.append(network_feed,obs[27:36])
            network_feed=np.append(network_feed,obs[49:51])
            states=torch.zeros(1,14, dtype=torch.float)
            for j in range(len(network_feed)):
                states[0][j]= network_feed[j]

            #Run grasp classifier


        plt.plot(x_angles, label="X Angle")
        plt.plot(y_angles, label="Y Angle")
        plt.plot(z_angles, label="Z Angle")
        plt.legend()
        plt.show()