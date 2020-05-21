import numpy as np
import gym
import pdb
from classifier_network import LinearNetwork, ReducedLinearNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import triang
#import serial
import matplotlib.pyplot as plt
import time

# take in data, make a change to th state of the arm (translate, rotate, or both)

def get_angles(local_obj_pos):
        obj_wrist = local_obj_pos[0:3]/np.linalg.norm(local_obj_pos[0:3])
        center_line = np.array([0,1,0])
        z_dot = np.dot(obj_wrist[0:2],center_line[0:2])
        z_angle = np.arccos(z_dot/np.linalg.norm(obj_wrist[0:2]))
        x_dot = np.dot(obj_wrist[1:3],center_line[1:3])
        x_angle = np.arccos(x_dot/np.linalg.norm(obj_wrist[1:3]))
        #print('angle calc took', t-time.time(), 'seconds')
        return x_angle,z_angle

def optimize_grasp(local_obs, init_reward,model):
    """
    try a bunch of different grasps and return the best one
    :param local_obs: initial starting coordinates in local frame
    :param init_reward: initial reward for initial grasp
    :return: full reward stack, best reward, coordinates for best reward
    """
    network_feed=local_obs[21:24]
    network_feed=np.append(network_feed,local_obs[25:34])
    local_obs=np.append(network_feed,local_obs[47:49])
    
    # x = _get_angles
    # obs = _get_obs()
    slide_step = 0.01
    joint_step = 0.2
    initial_obs = np.copy(local_obs)
    initial_reward = init_reward
    init_reward= init_reward.detach().numpy()
    init_reward=init_reward[0][0]
    iterations = 1000
    stored_obs = np.zeros(6)

    # try it and get a new classifier result
    # store it for us to play with
    # vary together
    for k in range(iterations):
        rand_delta = np.random.uniform(low=-slide_step, high=slide_step, size=3)
        rand_delta = np.append(rand_delta,np.random.uniform(low=-joint_step, high=joint_step, size=3))
        #print('local obs before',initial_obs)
        local_obs[0:6] = initial_obs[0:6] + rand_delta
        x_angle, z_angle = get_angles(local_obs[0:3]) # object location?
        local_obs[-2] = x_angle
        local_obs[-1] = z_angle
        #print('local obs after',local_obs)
        # feed into classifier
        states=torch.zeros(1,14, dtype=torch.float)
        for l in range(len(local_obs)):
            states[0][l]= local_obs[l]
        states=states.float()
        outputs = model(states)
        #print(outputs)
        outputs = outputs.detach().numpy()
        #print(type(outputs))
        #outputs = Grasp_net(inputs).cpu().data.numpy().flatten()
        reward_delta = outputs[0][0] - init_reward
        #print(reward_delta)
        rand_delta[0:3]=rand_delta[0:3]*20
        stored_obs += reward_delta / rand_delta[0:6]

    return stored_obs/np.linalg.norm(stored_obs)

# optimize_grasp(obs,init)









env = gym.make('gym_kinova_gripper:kinovagripper-v0')
env.reset()

env2 = gym.make('gym_kinova_gripper:kinovagripper-v0')
env2.reset()

model = ReducedLinearNetwork()
model=model.float()
model.load_state_dict(torch.load('trained_model_05_14_20_1349local.pt'))
model=model.float()
model.eval()
print('model loaded')

action_gradient = np.array([0,0.1,0,1,1,1]) # [9X1 normalized gradient of weights for actions]
ran_win = 1 / 2 # size of the window that random values are taken around
trial_num = 5 # number of random trials
action_size = 1 # should be same as Ameer's code action_size
step_size = 20 # number of actions taken by 
obs, reward, done, _= env.step([0,0,0,0,0,0])
network_feed=obs[21:24]
network_feed=np.append(network_feed,obs[25:34])
network_feed=np.append(network_feed,obs[47:49])
states=torch.zeros(1,14, dtype=torch.float)
for l in range(len(network_feed)):
    states[0][l]= network_feed[l]
states=states.float()
output = model(states)
action_gradient = optimize_grasp(obs,output, model)
print(action_gradient)
def sim_2_actions(ran_win, trial_num, action_size, step_size, action_gradient):
    action = np.zeros((trial_num,len(action_gradient)))
    new_rewards = np.zeros((trial_num))
    for i in range(trial_num):
        env2.reset()
        print('RESET')
        for j in range(len(action_gradient)):
            action[i][j] = action_size*np.random.uniform(action_gradient[j]+ran_win,action_gradient[j]-ran_win)
        for k in range(step_size):
            obs, reward, done, _ = env2.step(action[i,:])
            
            
            network_feed=obs[21:24]
            network_feed=np.append(network_feed,obs[25:34])
            network_feed=np.append(network_feed,obs[47:49])
            states=torch.zeros(1,14, dtype=torch.float)
            for l in range(len(network_feed)):
                states[0][l]= network_feed[l]
            states=states.float()
            output = model(states)
            #print(output)
            new_rewards[i] = output

    index = np.argmax(new_rewards)
    #print(action[index,:])
    #print('index',index)
    #print(np.max(new_rewards))
    #print('new rewards',new_rewards)
    for k in range(step_size):
        obs, reward, done, _= env.step(action[index,:])
        env.render()
        network_feed=obs[21:24]
        network_feed=np.append(network_feed,obs[25:34])
        network_feed=np.append(network_feed,obs[47:49])
        states=torch.zeros(1,14, dtype=torch.float)
        for l in range(len(network_feed)):
            states[0][l]= network_feed[l]
        states=states.float()
        output = model(states)
        print(output)
    



sim_2_actions(ran_win, trial_num, action_size, step_size, action_gradient)

