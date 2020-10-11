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
env = gym.make('gym_kinova_gripper:kinovagripper-v0')#,arm_or_end_effector="arm")
#print('action space',env.action_space.low, env.action_space.high)
#env.reset()
# setup serial
# ser = serial.Serial("/dev/ttyACM0", 9600)
# prev_action = [0.0,0.0,0.0,0.0]
action = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
t = 0
'''
size=[0.0175,0.02125,0.025]
xs=np.zeros([300,3])
ys=np.zeros([300,3])
colors = ('red', 'blue', 'green')
for j in range(3):
    for i in range(300):
        rand_x=triang.rvs(0.5)
        rand_x=(rand_x-0.5)*(0.16-2*size[j])
        rand_y=np.random.uniform()
        if rand_x>=0:
            rand_y=rand_y*(-(0.07-size[j]*np.sqrt(2))/(0.08-size[j])*rand_x+(0.07-size[j]*np.sqrt(2)))
        else:
            rand_y=rand_y*((0.07-size[j]*np.sqrt(2))/(0.08-size[j])*rand_x+(0.07-size[j]*np.sqrt(2)))
        xs[i,j]=rand_x
        ys[i,j]=rand_y
        
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(xs[:,0], ys[:,0], c='r')
ax.scatter(xs[:,1], ys[:,1], c='b')
plt.scatter(xs[:,2],ys[:,2],c='g')
plt.legend(['Small object','Medium Object','Large_object'])
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.axis([-0.09,0.09,-0.01,0.08])
plt.show()

'''
'''
model = ReducedLinearNetwork()
model=model.float()
model.load_state_dict(torch.load('trained_model_05_30_20_1119local.pt'))
model=model.float()
model.eval()
print('model loaded')
'''
coords='local'
episode_obs=[]
value=0
#t=time.time()
#ttot=np.array([])
for k in range(20):
    #env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    env.reset(start_pos=[0.03,0.01],obj_params=['Cube','B'],hand_orientation='not_random')
    x_move = np.random.rand()/10
    y_move = np.random.rand()/10
    #action = np.array([0.05-x_move,-y_move, 0.0, 0.0, 0.0, 0.0])
    #action= np.array([1, 1, 1, 0.3, 0.3, 0.3,0.1,0.1,0.1,0.1])
    #action= np.array([0, 0, 0.3,0.1,0.1,0.1])
    #print(0.5-x_move, -y_move)
    #env.randomize_initial_pos_data_collection()
    #env.save_vid()
    #print('reset')
    #t3=time.time()
    action=np.array([0,0,0,0.15, 0.3, 0.3])
    print('reset')
    for i in range(200):
        
        #print(action)
    # read action from pyserial
    # curr_action = ser.readline().decode('utf8').strip().split(",")
    # for i in range(4):
    #     curr_action[i] = float(curr_action[i])

    # if np.max(np.array(prev_action) - np.array(curr_action)) < 0.01:
    #     # keep going
    #     obs, reward, done, _ = env.step(prev_action)
    # else:
    #     # update action
    #     obs, reward, done, _ = env.step(curr_action)
        
        #if i == 70:
        #    print('move in x')
        #    action=np.array([0,0,0,0.3, 0.3, 0.3])
        #if i == 80:
        #    print('move in y')
        #    action=np.array([0,-0.1,0,0.0, 0.0, 0.0])
        if i == 150:
            print('move in z')
            action=np.array([0.15,0.05, 0.05, 0.05])
            env.env.pid=True

        '''
        if i ==10:
            action = np.array([0,0,0,0.3,0.3,0.3])
            
        if i ==100:
            action = np.array([0.0,0,0.15,0.3,0.3,0.3])
        '''
    # print((curr_action))
    # prev_action = curr_action
        if coords=='global':
            temp=np.array([action[0],action[1],action[2],1])
            #print(env.Twf,temp)
            action[0:3]=np.matmul(env.Twf[0:3,0:3],action[0:3])
            #action[3:6]=np.matmul(env.Twf[0:3,0:3],action[3:6])
        #print(action)
        env.render()
        obs, reward, done, _ = env.step(action)
        env.render()
        print(np.shape(obs))
        print(obs[-1])
        network_feed=obs[21:24]
            #print(np.shape(network_feed))
        #print('finger poses',np.matmul(env.env.Twf,[obs[0],obs[1],obs[2],1]))
        #print('joint states',env.env.Twf)
        network_feed=np.append(network_feed,obs[27:36])
        network_feed=np.append(network_feed,obs[49:51])
        states=torch.zeros(1,14, dtype=torch.float)
        #print(len(network_feed))
        for j in range(len(network_feed)):
            states[0][j]= network_feed[j]

        states=states.float()
        
    env.close()
print(value,'out of twenty')
with open('Training_Examples.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['obj x','obj y','obj z',"x slide","y slide","f1_prox", "f2_prox", "f3_prox", "f1_dist",'obj x len','obj y len','obj z len','x angle','z angle'])
    for i in range(len(episode_obs)):
        spamwriter.writerow(episode_obs[i])