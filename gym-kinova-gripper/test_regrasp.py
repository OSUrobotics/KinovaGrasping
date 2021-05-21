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
from scipy.spatial.transform import Slerp
#import serial
import matplotlib.pyplot as plt
#import optimizer
import csv
import time

def pid(kp, ki, kd, error, error_sum, ddterror):
    #if np.abs(error) <= .5:
    #    return 0
    kp_error = kp*error
    ki_error_sum = ki*error_sum
    kd_ddterror = kd*ddterror
    
    return kp_error + ki_error_sum + kd_ddterror, [kp_error, ki_error_sum, kd_ddterror]

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
actions=[[0,0,0,0.3,0.3,0.3, 0, 0, 0],[0,0,0,0.3,0.3,0,0,0, 0],[0,0,0,0,0,0.3,0,0,0], [0,0,0,0,0,0,.3,0,0], [0,0,0,0,0,0,0,.3,0], [0,0,0,0,0,0,0,0,.3]]
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

        error_sum_x = 0
        error_sum_y = 0
        error_sum_z = 0

        x_angles = []
        y_angles = []
        z_angles = []

        xpid_list = []
        ypid_list = []
        zpid_list = []

        
        #stem is X
        #dot product stem w/ [v1,v2,v3]=0 to get Y
        #   Set v1&v2 to random value
        #   solve for v3
        #Normalize new vec Y
        #cross product of X and Y to get Z
        #get rot matrix for world->stem [X,Y,Z]
        
        #Stem
        X = -np.array(vec)

        #Solve for v3
        v0,v1 =.1,.1
        v2 = (-(vec[0]*v0)-(vec[1]*v1))/vec[2]
        Y = np.array([v0,v1,v2]) / np.linalg.norm([v0,v1,v2])
        
        Z = np.cross(X,Y)
        #Stem to world
        Rsw = np.array([X,Y,Z])
        #print('RSW:',Rsw)
        #World to stem
        
        #Q_start = R.from_matrix(env.env.Twf[0:3,0:3]) - Twf at start of motion
        #Q_diff = (Q_current*Q_start).as_quat() - differene btwn current and start
        
        #Q_target = R.from_matrix(Rsw)
        #Q_current = R.from_matrix(env.env.Twf[0:3,0:3])
        #Q_r = (Q_target*Q_current).as_quat()
        
        #hand = env.env.sim.data.get_obj_qpos('j2s7s300_link_7') #hand in euler
        #stem = env.env.sim.data.get_obj_qpos('stem') #stem in euler
        #feed into slerp, convert to local? 
        
        
        hand = env.env._sim.data.get_body_xmat('j2s7s300_link_7')
        stem = env.env._sim.data.get_site_xmat('obj_vec')
        
        SLERP_TIMESTEPS = 200
        end_time = SLERP_TIMESTEPS*.04
        print(end_time)
        key_times=[0,end_time]
        
        slerp = Slerp(key_times,R.from_matrix([hand, stem]))
        interp_pts = slerp(np.arange(0, end_time, .04))
        euler =interp_pts.as_euler('xyz')
        print(len(euler))
        const_add = [0,0,0]
        
        #Timesteps
        for i in range(SLERP_TIMESTEPS+200):

            #want local_stem = [1,0,0]
            #local_stem = np.matmul(env.env.Tfw[0:3,0:3], vec)
            #print('local stem dir: ',local_stem)

            #Rotation calc: https://studywolf.wordpress.com/tag/orientation-control/
            #Local to world
            
            #print('Qt:', Q_target.as_quat())
            #print('Qc:', Q_current.as_quat())
            # calculate the rotation between current and target orientations
            #q_r = transformations.quaternion_multiply(q_target, transformations.quaternion_conjugate(q_e))            
            
            #if i > 0:
            #    prev_Q_r = Q_r.copy()
            
            #if i>0 and R.from_quat(Q_r-prev_Q_r).magnitude() > R.from_quat(-Q_r-prev_Q_r).magnitude():
            #    print("FLIPPING SIGN of Q_r")
            #    Q_r=-Q_r
            #print('Qr:', Q_r)
            #xcol = np.array([1,0,0]).reshape(3, 1)
            #mat_Q_r=R.from_quat(Q_r).as_matrix() 
            #print('Qr[1,0,0]T:\n', mat_Q_r.dot(xcol))
            #convert rotation quaternion to Euler angle forces            
            #u_task[3:] = ko * q_r[1:] * np.sign(q_r[0])

            #print('Qc xyz:', Q_current.as_euler('xyz'))
            #local_euler = # * np.sign(Q_r[3])
            #euler_angles = R.from_euler('xyz', local_euler).as_matrix()
            #euler_angles = R.from_matrix(np.matmul(euler_angles, env.env.Tfw[0:3,0:3])).as_euler('xyz')
            #print('global: ', euler_angles)
            #print('local: ', local_euler)
            # if i >= SLERP_TIMESTEPS:
            #     goal_euler = euler[-1]
            # else:
            #     goal_euler = euler[i]
    
            # if i > 0:
            #     diffs = goal_euler - prev_goal
            #     if abs(diffs[0]) >1.4:
            #         const_add[0]+=diffs[0]
            #     if abs(diffs[1]) >1.4:
            #         const_add[1]+=diffs[1]
            #     if abs(diffs[2]) >1.4:
            #         const_add[2]+=diffs[2]
    
            #Rt1w
            curr_angle = R.from_matrix(env.env._sim.data.get_body_xmat('j2s7s300_link_7')).inv()
            if i == 0:
                prev_angles = curr_angle
            # print('type(curr_angle):', type(curr_angle))
            # print('type(interp_pts[i]):', type(interp_pts[i]))
            # print('type(prev_angles):', type(prev_angles))
            #Rt1w * RwG1
            if i >= SLERP_TIMESTEPS:
                error = (curr_angle * interp_pts[-1]).as_euler('xyz')
            else:
                error = (curr_angle * interp_pts[i]).as_euler('xyz')
    
            #Rt1w * Rwt0            
            d_error = (curr_angle*(prev_angles.inv())).as_euler('xyz')

            x_angles.append(error[0])
            y_angles.append(error[1])
            z_angles.append(error[2])

                        #error_sum_x += goal_euler[0]-curr_euler[0]-const_add[0]
            #error_sum_y += goal_euler[1]-curr_euler[1]-const_add[1]
            #error_sum_z += goal_euler[2]-curr_euler[2]-const_add[2]

            #Successful w/yz .06, .04-.06
            #was goal_euler[0]-curr_euler[0]-const_add[0]
            rotx, xpid_vals = pid(.016, 0, 0.05, error[0], 0, -d_error[0])
            roty, ypid_vals = pid(.016, 0, 0.05, error[1], 0, -d_error[1])
            rotz, zpid_vals = pid(.008, 0, 0.05, error[2], 0, -d_error[2])
            
            prev_angles = curr_angle
            #prev_goal = goal_euler.copy()
            
            xpid_list.append(xpid_vals) 
            ypid_list.append(ypid_vals) 
            zpid_list.append(zpid_vals) 



            print(const_add)
            #Set action to lift
            #if i == 150:
            #    print('move in z')
            #    action=np.array([0.15,0.05, 0.05, 0.05])
            #    env.env.pid=True
            #    last_obs.append(obs)
            #testing

            action = np.array([0,0,0,0,0,0,rotx, roty, rotz])

            #print("diff pitch:", curr_pitch - Q_current.as_euler('xyz')[1])
            #action = np.array([0,0,0,0,0,0,local_euler[2]+curr_yaw,local_euler[1]+curr_pitch])                

            #if i == 25:
            #    action = np.array([0,0,0,0,0,0,0,0.3])

            #????
            #if coords=='global':
            #    temp=np.array([action[0],action[1],action[2],1])
            #    action[0:3]=np.matmul(env.Twf[0:3,0:3],action[0:3])
            
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

        plt.figure(1)
        plt.title("XYZ Angle Errors")
        plt.plot(x_angles, label="X Angle")
        plt.plot(y_angles, label="Y Angle")
        plt.plot(z_angles, label="Z Angle")
        plt.plot(euler[:,0], label="Goal X", linestyle="dotted")
        plt.plot(euler[:,1], label="Goal Y", linestyle="dotted")
        plt.plot(euler[:,2], label="Goal Z", linestyle="dotted")
        plt.legend()

        plt.figure(2)
        plt.title("X PID")
        xpid_list = np.array(xpid_list)
        plt.plot(xpid_list[:,0], label="P")
        plt.plot(xpid_list[:,1], label="I")
        plt.plot(xpid_list[:,2], label="D")
        plt.legend()

        plt.figure(3)
        plt.title("Y PID")
        ypid_list = np.array(ypid_list)
        plt.plot(ypid_list[:,0], label="P")
        plt.plot(ypid_list[:,1], label="I")
        plt.plot(ypid_list[:,2], label="D")
        plt.legend()

        plt.figure(4)
        plt.title("Z PID")
        zpid_list = np.array(zpid_list)
        plt.plot(zpid_list[:,0], label="P")
        plt.plot(zpid_list[:,1], label="I")
        plt.plot(zpid_list[:,2], label="D")
        plt.legend()
        plt.show()
