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
import optimizer
import csv
import time
env = gym.make('gym_kinova_gripper:kinovagripper-v0')
env.reset()
# setup serial
# ser = serial.Serial("/dev/ttyACM0", 9600)
# prev_action = [0.0,0.0,0.0,0.0]
action = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0])
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
model = ReducedLinearNetwork()
model=model.float()
model.load_state_dict(torch.load('trained_model_05_14_20_1349local.pt'))
model=model.float()
model.eval()
print('model loaded')
episode_obs=[]
#t=time.time()
#ttot=np.array([])
for k in range(20):
    #env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    env.reset()
    x_move = np.random.rand()/10
    y_move = np.random.rand()/10
    #action = np.array([0.05-x_move,-y_move, 0.0, 0.0, 0.0, 0.0])
    action= np.array([0, 0, 0.0, 0.0, 0.0, 0.0])
    #print(0.5-x_move, -y_move)
    #env.randomize_initial_pos_data_collection()
    #env.save_vid()
    #print('reset')
    #t3=time.time()
    for i in range(250):
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
        
        if i == 70:
            print('move in x')
            action=np.array([0.1,0,0,0.0, 0.0, 0.0])
        if i == 100:
            print('move in y')
            action=np.array([0,0.1,0,0.0, 0.0, 0.0])
        if i == 170:
            print('move in z')
            action=np.array([0,0,0.1,0, 0.0, 0.0])
        
        '''
        if i ==10:
            action = np.array([0,0,0,0.3,0.3,0.3])
            
        if i ==100:
            action = np.array([0.0,0,0.15,0.3,0.3,0.3])
        '''
    # print((curr_action))
    # prev_action = curr_action
        obs, reward, done, _ = env.step(action)
        #print(optimizer.optimize_grasp(obs,reward))
        #print(obs)
        #obs=np.copy(obs)
        network_feed=obs[21:24]
            #print(np.shape(network_feed))
        network_feed=np.append(network_feed,obs[25:34])
        network_feed=np.append(network_feed,obs[47:49])
        #print('network feed', network_feed)
        states=torch.zeros(1,14, dtype=torch.float)
        for j in range(len(network_feed)):
            states[0][j]= network_feed[j]
        #print(states)
        #input_stuff=torch.tensor(network_feed,dtype=torch.float)
        
        states=states.float()
        env.render()
        #t3=time.time()
        #np.random.rand(14)
        output = model(states)
        #t4=time.time()
        
        #ttot=np.append(ttot,t4-t3)
        #print(output)
        #if output >0.2:
        #    if k ==0:
        #        episode_obs=np.copy(network_feed)
        #    else:
        #        episode_obs=np.vstack((episode_obs,network_feed))
        #    break
    #print(episode_obs)
        #network_feed=obs[0:5]
        #print('Distal 1,', obs[9:12])
        #print('Distal 1,', obs[12:15])
        #print('Distal 1,', obs[15:18])
        #print('Wrist,', obs[18:21])
        #network_feed=np.append(network_feed,obs[6:23])
        #network_feed=np.append(network_feed,obs[24:])
        #input_stuff=torch.tensor(network_feed,dtype=torch.float)
        #print(input_stuff)
        #print(model(input_stuff))
        #print(done)
    # action[1] += 0.5
    # action[2] += 0.2
    # action[3] += 0.7
        #env.render()
    # if t > 25:
    #     action = np.array([0.1, 0.8, 0.8, 0.8])
    # # print()
    # t += 1
    #t4=time.time()
    #print(t3-t4)
#t2=time.time()
#print('it took ',t2-t,'seconds')
#print(np.sum(ttot))
with open('Training_Examples.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['obj x','obj y','obj z',"x slide","y slide","f1_prox", "f2_prox", "f3_prox", "f1_dist",'obj x len','obj y len','obj z len','x angle','z angle'])
    for i in range(len(episode_obs)):
        spamwriter.writerow(episode_obs[i])