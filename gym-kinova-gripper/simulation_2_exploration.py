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
<<<<<<< HEAD
        return x_angle, z_angle
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def optimize_grasp(local_obs, init_reward, model):
    """ 
    try a bunch of different grasps and return the best one
    :param local_obs: initial starting coordinates in local frame
    :param init_reward: initial reward for initial grasp
    :param model: network model
    :return: full reward stack, best reward, coordinates for best reward
    """
    network_feed=local_obs[21:24]
    network_feed=np.append(network_feed,local_obs[27:36])
    local_obs=np.append(network_feed,local_obs[49:51])
    #network_feed=obs[0:61]
    use_gc=True
    
    slide_step = 0.05
    joint_step = 0.5
    initial_obs = np.copy(local_obs)
    # initial_reward = init_reward

    if use_gc:
        init_reward= init_reward.detach().numpy()
        init_reward = init_reward[0][0]
        temp=np.linalg.norm(local_obs[0:3])
        init_reward+=np.sum(local_obs[3:6])/25*np.sign(0.1-temp)
        init_reward += (0.08-temp)
        init_reward=sigmoid(init_reward)
    else:
        init_reward=0
        #print('starting init reward', init_reward)
        #testing adding some fancy stuff
        temp=np.linalg.norm(local_obs[0:3])
        init_finger_reward=np.sum(local_obs[3:6])/5
        init_slide_reward = init_reward +(0.08-temp)*5
        #print('modified init reward', init_reward)
        init_finger_reward=sigmoid(init_finger_reward)
        init_slide_reward=sigmoid(init_slide_reward)

=======
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
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
    iterations = 1000
    stored_obs = np.zeros(6)

    # try it and get a new classifier result
    # store it for us to play with
    # vary together
    for k in range(iterations):
        rand_delta = np.random.uniform(low=-slide_step, high=slide_step, size=3)
<<<<<<< HEAD
        rand_finger = np.random.uniform(low=-joint_step, high=joint_step, size=3)
        # rand_finger = np.array([rand_finger, rand_finger, rand_finger])
        rand_delta = np.append(rand_delta, rand_finger)
        # rand_delta = np.append(rand_delta,np.random.uniform(low=-joint_step, high=joint_step, size=3))

        local_obs[0:6] = initial_obs[0:6] + rand_delta
        local_obs[2]= initial_obs[2]
        local_obs[-2], local_obs[-1] = get_angles(local_obs[0:3])
        # feed into classifier
        states=torch.zeros(1,14, dtype=torch.float)
        #print(local_obs)
        for l in range(len(local_obs)):
            states[0][l]= local_obs[l]
        states=states.float()

        #print(type(outputs))
        #outputs = Grasp_net(inputs).cpu().data.numpy().flatten()
        outputs = model(states)
        outputs = outputs.detach().numpy()
        if use_gc:
            temp=np.linalg.norm(local_obs[0:3])
            reward=outputs[0][0]
            reward+=np.sum(local_obs[3:6])/25*np.sign(0.125-temp)
            reward +=(0.08-temp)
            #print('pre sigmoid',reward)
            reward = sigmoid(reward)
            #print('post sigmoid',reward)
            reward_delta = reward-init_reward
            gradient_delta=rand_delta
            gradient_delta[0:3]=-gradient_delta[0:3]
            stored_obs +=reward_delta*reward*(1-reward)*gradient_delta[0:6]
        else:
            temp=np.linalg.norm(local_obs[0:3])
            finger_reward=np.sum(local_obs[3:6])/5
            slide_reward = outputs[0][0] +(0.08-temp)*5
            slide_reward=sigmoid(slide_reward)
            finger_reward=sigmoid(finger_reward)
            #print('iteration reward is',iteration_reward)
            #print('finger reward is',finger_reward)
            #print('output was ', output[0][0], 'it is now', iteration_reward)
            slide_reward_delta = slide_reward - init_slide_reward
            finger_reward_delta = finger_reward - init_finger_reward
            #print(slide_reward_delta, 'reward for an action of ', rand_delta[0:3])
            #rand_delta[0:3]=rand_delta[0:3]*joint_step/slide_step
            stored_obs[0:3] += (-slide_reward_delta)*(slide_reward)*(1-slide_reward)*(rand_delta[0:3])
            stored_obs[3:6] += finger_reward_delta *finger_reward*(1-finger_reward)*(rand_delta[3:6])
    #print('final count of better grasps is ',f)
    return stored_obs/np.linalg.norm(stored_obs)

def sim_2_actions(ran_win, trial_num, action_size, step_size, action_gradient,og_obs):
    """
    use action gradient to take steps and test new grasp
    """
    blind_action=[0,0,0,0.3,0.3,0.3]
    '''
    action = np.zeros((trial_num,len(action_gradient)))
    new_rewards = np.zeros((trial_num))
    for i in range(trial_num):
        #reset_stuff=og_obs[24:30]
        qpos=env.get_sim_state()
        reset_stuff=og_obs[21:24]
        env2.reset(start_pos=reset_stuff,coords='local',qpos=qpos,obj_params=[shape[fuck],size[this]])
        #print('RESET')

=======
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
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
        for j in range(len(action_gradient)):
            action[i][j] = action_size*np.random.uniform(action_gradient[j]+ran_win,action_gradient[j]-ran_win)
        for k in range(step_size):
            obs, reward, done, _ = env2.step(action[i,:])
<<<<<<< HEAD
            #env2.render()
            
            network_feed=obs[21:24]
            network_feed=np.append(network_feed,obs[27:36])
            network_feed=np.append(network_feed,obs[49:51])
            #network_feed = obs[0:61]

=======
            
            
            network_feed=obs[21:24]
            network_feed=np.append(network_feed,obs[25:34])
            network_feed=np.append(network_feed,obs[47:49])
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
            states=torch.zeros(1,14, dtype=torch.float)
            for l in range(len(network_feed)):
                states[0][l]= network_feed[l]
            states=states.float()
            output = model(states)
            #print(output)
            new_rewards[i] = output
<<<<<<< HEAD
    '''

    action=action_gradient
    #index = np.argmax(new_rewards)

=======

    index = np.argmax(new_rewards)
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
    #print(action[index,:])
    #print('index',index)
    #print(np.max(new_rewards))
    #print('new rewards',new_rewards)
    for k in range(step_size):
<<<<<<< HEAD
        #print(action[index,:])
        obs, reward, done, _= env.step(action)
        env3.step(blind_action)
        #env.render()
        network_feed=obs[21:24]
        network_feed=np.append(network_feed,obs[27:36])
        network_feed=np.append(network_feed,obs[49:51])
        #network_feed=obs[0:61]

=======
        obs, reward, done, _= env.step(action[index,:])
        env.render()
        network_feed=obs[21:24]
        network_feed=np.append(network_feed,obs[25:34])
        network_feed=np.append(network_feed,obs[47:49])
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47
        states=torch.zeros(1,14, dtype=torch.float)
        for l in range(len(network_feed)):
            states[0][l]= network_feed[l]
        states=states.float()
        output = model(states)
<<<<<<< HEAD

        #print("new reward:", output[0])
    return obs

env = gym.make('gym_kinova_gripper:kinovagripper-v0') #this is the actual one AKA the real world
env.reset()

env2 = gym.make('gym_kinova_gripper:kinovagripper-v0') #this is the fake-o
env2.reset()

env3 = gym.make('gym_kinova_gripper:kinovagripper-v0') #this is the blind-o
env3.reset()

our_score=[]
blind_score=[]
model = ReducedLinearNetwork()
model=model.float()
model.load_state_dict(torch.load('trained_model_05_31_20_2208local.pt'))
model=model.float()
model.eval()
print('model loaded')
shape=['Cylinder','Box']
size=['B','M','S']
gradient_processing_times=[]
sim_times=[]
all_times=[]
for fuck in range(2):
    for this in range(3):
        our_score=[]
        for k in range(100):
            if k%20==0:
                print(k)
            env.reset(obj_params=[shape[fuck],size[this]])
            env2.reset(obj_params=[shape[fuck],size[this]])
            env3.reset(obj_params=[shape[fuck],size[this]])
            action_gradient = np.array([0,0.1,0,1,1,1]) # [9X1 normalized gradient of weights for actions]
            ran_win = 0.005 # size of the window that random values are taken around
            trial_num = 5 # number of random trials
            action_size = 1 # should be same as Ameer's code action_size
            step_size = 6 # number of actions taken by 
            obs, reward, done, _= env.step([0,0,0,0,0,0])
            obs2, reward2, done2, _ = env2.step([0,0,0,0,0,0]) 
            network_feed=obs[21:24]
            #print('object position',obs[21:24])
            network_feed=np.append(network_feed,obs[27:36])
            network_feed=np.append(network_feed,obs[49:51])
            #network_feed = obs[0:61]
            states=torch.zeros(1,14, dtype=torch.float)
            for l in range(len(network_feed)):
                states[0][l]= network_feed[l]
            states=states.float()
            output = model(states)
            a=True
            for i in range(50):
                x_move = np.random.rand()/4
                y_move = np.random.rand()/4
                action = np.array([0.125-x_move,0.125-y_move, 0.0, 0.3, 0.3, 0.3])
                obs, reward, done, _=env.step(action)
                env2.step(action)
                network_feed=obs[21:24]
                network_feed=np.append(network_feed,obs[27:36])
                network_feed=np.append(network_feed,obs[49:51])
                #network_feed=obs[0:61]
                states=torch.zeros(1,14, dtype=torch.float)
                for l in range(len(network_feed)):
                    states[0][l]= network_feed[l]
                states=states.float()
                output = model(states)
                #print('reward is,', output)
                #env.render()
            break_next=False        
            reset_stuff=obs[21:24]
            qpos=env.get_sim_state()
            env3.reset(start_pos=reset_stuff,coords='local',qpos=qpos,obj_params=[shape[fuck],size[this]])
            
            for i in range(8):
                #print("initial_reward:", output[0])
                t=time.time()
                action_gradient = optimize_grasp(obs,output, model)
                t2=time.time()
                #print("action_boi:: ", action_gradient)
                obs=sim_2_actions(ran_win, trial_num, action_size, step_size, action_gradient,obs)
                
                network_feed=obs[21:24]
                network_feed=np.append(network_feed,obs[27:36])
                network_feed=np.append(network_feed,obs[49:51])
                #network_feed=obs[0:61]
                states=torch.zeros(1,14, dtype=torch.float)
                for l in range(len(network_feed)):
                    states[0][l]= network_feed[l]
                states=states.float()
                output = model(states)
                #print('classifier says that the grasp is ,', output)
                if break_next:
                    break
                if output > 0.9:
                    break
                    break_next=True
                t3=time.time()
                sim_times.append(t3-t2)
                all_times.append(t3-t)
                gradient_processing_times.append(t2-t)
                
                
            action = [0,0,0.15,0.1,0.1,0.1]
            for i in range(100):
                #print('we are here now, the next ones are the env')
                obs,reward,done,_ = env.step(action)
                #print('the next ones are the env3')
                _, blind_reward, blind_done, _= env3.step(action)
                #print('anything after this is rendering')
                #env.render()
                #print('after this is the check for done and blind done')
                if done and blind_done:
                    break
                #print('nothing should happen after this and before the "we are here now..." line')
            blind_score=np.append(blind_score,blind_reward)
            our_score=np.append(our_score,reward)
        print(shape[fuck],size[this],'our score', sum(our_score), 'vs blind score', sum(blind_score))
        print('average of gradient times',np.average(gradient_processing_times), '+/-', np.std(gradient_processing_times))
        print('average of sim times',np.average(sim_times), '+/-', np.std(sim_times))
        print('average of both times',np.average(all_times), '+/-', np.std(all_times))
=======
        print(output)
    



sim_2_actions(ran_win, trial_num, action_size, step_size, action_gradient)
>>>>>>> 38e1e2f0ef791a285691a13b0a5816ff9ae64f47

