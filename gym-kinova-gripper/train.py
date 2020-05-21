#!/usr/bin/env python3

'''
Author : Yi Herng Ong
Date : 9/29/2019
Purpose : Supervised learning for near contact grasping strategy
'''


import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import TD3
import gym
import utils
import argparse
import torch.optim as optim
import pdb
import pickle
import datetime
import NCS_nn
# import expert_data
import random
import pandas 
from ounoise import OUNoise
from classifier_network import LinearNetwork, ReducedLinearNetwork
from trainGP import trainGP
import matplotlib.pyplot as plt
import csv



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PID(target, current):
    err = target - current
    if err < 0.0:
        err = 0.0
    diff = err / 4
    vel = err + diff # PD control
    action = (vel / 0.8) * 0.3 
    
    return action

def getRandomVelocity():
    # if not flag:
    flag = np.random.choice(np.array([1, 0]), p = [0.5, 0.5])
    if flag:
        f1 = np.random.uniform(0.0, 0.3)
        f2 = np.random.uniform(0.0, 0.3)
        f3 = np.random.uniform(0.0, 0.3)
    else:
        f1 = np.random.uniform(-0.3, 0.3)
        f2 = np.random.uniform(-0.3, 0.3)
        f3 = np.random.uniform(-0.3, 0.3)        
    vels = np.array([f1, f2, f3])
    return vels

def normalize_vector(vector):
    #print(vector-np.min(vector))
    #print(np.max(vector)-np.min(vector))
    if (np.max(vector)-np.min(vector)) == 0:
        n_vector=np.ones(np.shape(vector))*0.5
    else:
        n_vector=(vector-np.min(vector))/(np.max(vector)-np.min(vector))
    return n_vector

def convert_to_local(data):
    adjustment=data[:,18:21]
    #print(adjustment)
    local_data=data
    for i in range(24):
        #print(i)
        local_data[:,i]=data[:,i]-adjustment[:,i%3]
    return local_data

def test(env, trained_model):
    actor_net = NCS_nn.NCS_net(48, 4, 0.8).to(device)
    model = torch.load(trained_model)
    actor_net.load_state_dict(model)
    actor_net.eval()

    # IF YOU WAN TO START AT RANDOM INTERMEDIATE STATE
    # file_name = open("data_cube_5_10_07_19_1612.pkl", "rb")
    # data = pickle.load(file_name)
    # states = np.array(data["states"])
    # random_states_index = np.random.randint(0, len(states), size = len(states))

    noise = OUNoise(4)
    expl_noise = OUNoise(4, sigma=0.001)
    for _ in range(10):
        # inference
        obs, done = env.reset(), False
        # obs = env.env.intermediate_state_reset(states[np.random.choice(random_states_index, 1)[0]])
        print("start")
        # while not done:
        for _ in range(150):
            obs = torch.FloatTensor(np.array(obs).reshape(1,-1)).to(device) # + expl_noise.noise()
            action = actor_net(obs).cpu().data.numpy().flatten()
            print(action)
            obs, reward, done, _ = env.step(action)
            # print(reward)

# def train_network(data_filename, max_action, num_epoch, total_steps, batch_size, model_path="trained_model"):
def train_network(training_set, training_label, num_epoch, total_steps, batch_size, all_testing_set,all_testing_label,model_path="trained_model"):
    # import data
    # file = open(data_filename + ".pkl", "rb")
    # data = pickle.load(file)
    # file.close()

    ##### Training Action Net ######
    # state_input = data["states"]
    # actions = data["label"]
    # actor_net = NCS_nn.NCS_net(len(state_input[0]), len(actions[0]), max_action).to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)

    ##### Training Grasp Classifier ######
    # state_input = data[:, 0:-1]
    # state_input = data["states"]
    # actions = data["grasp_success"]
    # total_steps = data["total_steps"]
    # actions = data[:, -1]
    # pdb.set_trace()
    classifier_net = ReducedLinearNetwork()
    classifier_net=classifier_net.float()
    
    '''
    actor_net = NCS_nn.GraspValid_net(len(state_input[0])).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)
    '''
    
    total_percent=np.zeros([int(num_epoch/10),2])
    true_pos=np.zeros([int(num_epoch/10),6,2])
    false_pos=np.zeros([int(num_epoch/10),6,2])
    num_update = total_steps / batch_size
    
    # print(num_update, len(state_input[0]), len(actions))

    for epoch in range(num_epoch):
            # actions_all_loc = np.random.randint(0,total_steps, size=total_steps)
            # np.random.shuffle(actions_all_loc)
            # actions_all_loc = np.array(actions_all_loc)
            #if epoch%10==0:
            #    total_percent[epoch/10,0],true_pos[epoch/10,:,0],false_pos[epoch/10,:,0]=test_network(all_testing_set, all_testing_label,classifier_net,0.9)
            #    total_percent[epoch/10,1],true_pos[epoch/10,:,1],false_pos[epoch/10,:,1]=test_network(all_testing_set, all_testing_label,classifier_net,0.9)
            #shuffle the data before it gets fed into the network
            num_data_points=np.shape(training_set)
            #print(num_data_points)
            #print(num_data_points)
            random_indicies=np.arange(num_data_points[0])
            #print(random_indicies)
            #print(random_indicies)
            np.random.shuffle(random_indicies)
            #print(random_indicies)
            #print(random_indicies)
            training_set=training_set[random_indicies,:]
            training_label=training_label[random_indicies]
            network_feed=training_set[:,21:24]
            #print(np.shape(network_feed))
            network_feed=np.append(network_feed,training_set[:,27:36], axis=1)
            network_feed=np.append(network_feed,training_set[:,49:51], axis=1)
            #print(np.shape(network_feed))
            running_loss = 0.0
            start_batch = 0
            end_batch = batch_size
            learning_rate=0.1-epoch/num_epoch*0.09
            for i in range(int(num_update)):
                # zero parameter gradients
                classifier_net.zero_grad()
                # forward, backward, and optimize
                #ind = np.arange(start_batch, end_batch)
                start_batch += batch_size
                end_batch += batch_size
                # states = torch.FloatTensor(np.array(state_input)[ind]).to(device)
                # labels = torch.FloatTensor(np.array(actions)[ind].reshape(-1, 1)).to(device)

                states = torch.tensor(network_feed[i])
                labels = torch.tensor(training_label[i])
                states=states.float()
                labels=labels.float()
                # labels = torch.FloatTensor(np.array(training_label)[ind].reshape(-1, 1)).to(device)                
                # labels = torch.FloatTensor(np.array(actions)[ind]).to(device)

                output = classifier_net(states)
                #output=output.reshape(1)
                # pdb.set_trace()
                criterion = nn.MSELoss()
                loss = criterion(output, labels)
                classifier_net.zero_grad()
                loss.backward()
                for f in classifier_net.parameters():
                    f.data.sub_(f.grad.data * learning_rate)
                running_loss += loss.item() 
                # print("loss", loss.item())
                if (i % 100) == 99:
                    print("Epoch {} , idx {}, loss: {}".format(epoch + 1, i + 1, running_loss/(100)))
                    running_loss = 0.0
    print("Finish training, saving...")
    print('Percent Correct ',total_percent)
    print('False Positives ',false_pos)
    print('True Positives ',true_pos)
    torch.save(classifier_net.state_dict(), model_path + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + "local" + ".pt")    
    return classifier_net, total_percent, false_pos, true_pos

def test_network(test_in,test_out,classifier_net,output_threshold=0.8):
    num_tests=np.shape(test_in)
    #print(num_tests)
    test_in=convert_to_local(test_in)
    num_labels=np.array([200,200,200,200,200,200])
    num_true=np.zeros(6)
    num_false=np.zeros(6)
    percent_correct=np.array([0,0,0,0,0,0],dtype='f')
    num_false_pos=np.array([0,0,0,0,0,0],dtype='f')
    num_false_neg=np.array([0,0,0,0,0,0],dtype='f')
    num_true_pos=np.array([0,0,0,0,0,0],dtype='f')
    past_vals=0
    false_pos_indicies=[]
    false_neg_indicies=[]
    #for i in range(6):
    i=0
    '''
    network_feed=test_in[:,0:5]
    #print(np.shape(network_feed))
    network_feed=np.append(network_feed,test_in[:,6:23], axis=1)
    network_feed=np.append(network_feed,test_in[:,24:], axis=1)
    '''
    
    network_feed=test_in[:,21:24]
            #print(np.shape(network_feed))
    network_feed=np.append(network_feed,test_in[:,27:36], axis=1)
    network_feed=np.append(network_feed,test_in[:,49:51], axis=1)
    for j in range(num_tests[0]):
        #print(test_out[j+past_vals])
        #states=torch.tensor(test_in[j+past_vals])
        #labels=torch.tensor(test_out[j+past_vals])
        
        states=torch.tensor(network_feed[j])
        labels=torch.tensor(test_out[j])
        states=states.float()
        labels=labels.float()
        output=classifier_net(states)
        #print(output)
        if output > output_threshold:
            net_out=1
            #print('1')
        else:
            net_out=0
            #print('0')
        #print(labels.item())
        #print(abs(net_out-labels.item()))
        percent_correct[i]=percent_correct[i]+abs(net_out-labels.item())
        #print(percent_correct[i])
        if (net_out-labels.item())==1:
            num_false_pos[i]=num_false_pos[i]+1
            false_pos_indicies.append(j+past_vals)
        elif (net_out-labels.item())==-1:
            num_false_neg[i]=num_false_neg[i]+1
            false_neg_indicies.append(j+past_vals)
        elif (net_out>=0.5) & (labels.item() >=0.5):
            num_true_pos[i]=num_true_pos[i]+1
            #print(percent_correct)
        #print(num_labels)
        #print(percent_correct)
        #num_true[i]=np.count_nonzero(test_out[past_vals:j+past_vals]>0.5)
        #num_false[i]=np.count_nonzero(test_out[past_vals:j+past_vals]<0.5)
    num_true[i]=np.count_nonzero(test_out[:]>0.5)
    num_false[i]=np.count_nonzero(test_out[:]<=0.5)
    percent_correct[i]=1-percent_correct[i]/num_tests[0]
    past_vals=past_vals+num_labels[i]

    print('correctly identified grasps ', percent_correct[i]*100, '% of the time')
    print('false positives', num_false_pos)
    print('false negatives', num_false_neg)
    print(num_true, num_false)
    true_pos_rate=num_true_pos/num_true
    false_pos_rate=num_false_pos/num_false
    total_percent=sum(percent_correct)
    return total_percent, true_pos_rate, false_pos_rate, false_pos_indicies, false_neg_indicies
    

def get_false_grasps(false_pos_indicies, false_neg_indicies, dataset):
    print('im working on it, ok?!')
    print(false_neg_indicies)
    print(np.shape(dataset))
    num_pos=len(false_pos_indicies)
    num_neg=len(false_neg_indicies)
    neg_ones=np.zeros([num_neg,66])
    pos_ones=np.zeros([num_pos,66])
    for i in range(num_pos):
        pos_ones[i] = dataset[false_pos_indicies[i]]
    for i in range(num_neg):
        neg_ones[i] = dataset[false_neg_indicies[i]]
    if num_pos > 10:
        num_pos = 10
    if num_neg > 10:
        num_neg = 10    
        
    np.random.shuffle(pos_ones)
    np.random.shuffle(neg_ones)
    pos_ones=pos_ones[0:num_pos,:]
    neg_ones=neg_ones[0:num_neg,:]
    pos_ones = np.concatenate((pos_ones[:,21:24],pos_ones[:,25:28]),axis=1)
    neg_ones = np.concatenate((neg_ones[:,21:24],neg_ones[:,25:28]),axis=1)
    return pos_ones, neg_ones

    
def show_false_grasps(pos_ones,neg_ones, obj_shape, obj_size):
    env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    #start = time.time()
    
    model = LinearNetwork()
    model.load_state_dict(torch.load('trained_model_04_22_20_1727.pt'))
    model.eval()
    '''
    graspSuccess_label = []
    obs_label = []
    a=np.shape(pos_ones)
    for episode in range(a[1]):
        obs, done = env.reset(start_pos=pos_ones[i,0:3], obj_params=[obj_shape,obj_size]), False
        #print(obs[21:24])
        reward = 0
        target_joint_config = pos_ones[i,3:]
        step = 0
        prelim_step=0
        reach = False
        episode_obs_label = []
        while not done:
            # finger action 
            finger_action = []
            # First part : go to random joint config         
            # print((np.array(obs[25:28])))
            if (np.max(np.abs((np.array(obs[25:28]) - target_joint_config))) > 0.01) and (reach == False) and prelim_step <200:
                for finger in range(3):
                    # print(target_joint_config[i], obs[25+i])
                    finger_action.append(PID(target_joint_config[finger], obs[25+finger]))
                action = np.array([0.0, finger_action[0], finger_action[1], finger_action[2]])
                episode_obs_label=obs
                prelim_step+=1
            # Second part : close fingers
            else:
                reach = True # for not going back to the previous if loop
                step += 1    
                if step ==1:
                    env.pause()
                    print('we got here with finger position difference', np.abs((np.array(obs[25:28]) - target_joint_config)), 'and object position difference, ',np.abs((np.array(obs[21:24]) - pos_ones[i,0:3])) )
                if step > 25:
                    # finger_action = getRandomVelocity()
                    action = np.array([0.3, 0.05, 0.05, 0.05])
                    #print('lifting')
                else:
                    action = np.array([0.0, 0.0, 0.0, 0.0])

            #print(step)
            #print('data collection pre', done)
            obs, reward, done, _ = env.step(action)
            #print(obs[-1])
            #print('data collection', done)
            env.render()
            network_feed=obs[0:5]
            #print(obs[21:24])
            network_feed=np.append(network_feed,obs[6:23])
            network_feed=np.append(network_feed,obs[24:])
            input_stuff=torch.tensor(network_feed,dtype=torch.float)
            #print(input_stuff)
            print(model(input_stuff))
            #print(model(obs))

        # If object is lifted,     
        if reward:
            graspSuccess_label.append(1)
        else:
            graspSuccess_label.append(0)
        obs_label.append(episode_obs_label)
        #print(obs_label)
        #Sprint(graspSuccess_label)
        #print(obs_label)
        print(episode)
    # print(time.time() - start)
    # pdb.set_trace()
    '''
    b=np.shape(neg_ones)
    for episode in range(b[1]):
        obs, done = env.reset(start_pos=neg_ones[i,0:3], obj_params=[obj_shape,obj_size]), False
        #print(obs[21:24])
        reward = 0
        target_joint_config = neg_ones[i,3:]
        step = 0
        prelim_step=0
        reach = False
        episode_obs_label = []
        #random_finger_action = getRandomVelocity()
        while not done:
            # finger action 
            finger_action = []
            # First part : go to random joint config         
            # print((np.array(obs[25:28])))
            if (np.max(np.abs((np.array(obs[25:28]) - target_joint_config))) > 0.01) and (reach == False) and prelim_step <200:
                for finger in range(3):
                    # print(target_joint_config[i], obs[25+i])
                    finger_action.append(PID(target_joint_config[finger], obs[25+finger]))
                action = np.array([0.0, finger_action[0], finger_action[1], finger_action[2]])
                episode_obs_label=obs
                prelim_step+=1
            # Second part : close fingers
            else:
                reach = True # for not going back to the previous if loop
                step += 1     
                if step ==1:
                    env.pause()
                    print('we got here with finger position difference', np.abs((np.array(obs[25:28]) - target_joint_config)), 'and object position difference, ',np.abs((np.array(obs[21:24]) - neg_ones[i,0:3])) )

                if step >= 25: # wait for one second
                    action = np.array([0.3, 0.05, 0.05, 0.05])
                    #print('lifting')
                else:
                    action = np.array([0.0, 0.0, 0.0, 0.0])

            #print(step)
            #print('data collection pre', done)
            obs, reward, done, _ = env.step(action)
            #print(obs[-1])
            #print('data collection', done)
            env.render()
            network_feed=obs[0:5]
            #print(obs[21:24])
            network_feed=np.append(network_feed,obs[6:23])
            network_feed=np.append(network_feed,obs[24:])
            input_stuff=torch.tensor(network_feed,dtype=torch.float)
            #print(input_stuff)
            print(model(input_stuff))
            #print(model(obs))

        # If object is lifted,     
        if reward:
            graspSuccess_label.append(1)
        else:
            graspSuccess_label.append(0)
        obs_label.append(episode_obs_label)
        #print(obs_label)
        #Sprint(graspSuccess_label)
        #print(obs_label)
        print(episode)
    # print(time.time() - start)
    # pdb.set_trace()


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0")    # OpenAI gym environment name

    # parser.add_argument("--num_episode", default=1e4, type=int)                            # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--batch_size", default=250, type=int)                            # Batch size for updating network
    # parser.add_argument("--epoch", default=20, type=int)                                    # number of epoch

    # parser.add_argument("--data_gen", default=0, type=int)                                # bool for whether or not to generate data
    # parser.add_argument("--data", default="data" )                                        # filename of dataset (the entire traj)
    # parser.add_argument("--grasp_success_data", default="grasp_success_data" )            # filename of grasp success dataset    
    # parser.add_argument("--train", default=1, type=int)                                    # bool for whether or not to train data
    # parser.add_argument("--model", default="model" )                                    # filename of model for training    
    # parser.add_argument("--trained_model", default="trained_model" )                    # filename of saved model for testing
    # parser.add_argument("--collect_grasp", default=0, type=int )                        # check to collect either lift data or grasp data
    # parser.add_argument("--grasp_total_steps", default=100000, type=int )                # number of steps that need to collect grasp data

    # # dataset_cube_2_grasp_10_04_19_1237
    # args = parser.parse_args()

    # if args.data_gen:
    #     env = gym.make(args.env_name)
    #     state_dim = env.observation_space.shape[0]
    #     action_dim = env.action_space.shape[0] 
    #     max_action = env.action_space.high # action needs to be symmetric
    #     max_steps = 100 # changes with env
    #     if args.collect_grasp == 0:
    #         data = expert_data.generate_Data(env, args.num_episode, args.data)
    #     else:
    #         # print("Here")
    #         data = expert_data.generate_lifting_data(env, args.grasp_total_steps, args.data, args.grasp_success_data)
    # else:

    #     if args.train:
    #         assert os.path.exists(args.data + ".pkl"), "Dataset file does not exist"
    #         # actor_net = train_network(args.data, actor_net, args.epoch, args.num_episode*max_steps, args.batch_size, args.model)
    #         actor_net = train_network(args.data, 0.8, args.epoch, 400000, args.batch_size, args.model)
        
    #     else:
    #         # assert os.path.exists(args.data + ".pkl"), "Dataset file does not exist"
    #         env = gym.make(args.env_name)
    #         test(env, args.trained_model)

    # data_filename = "/home/graspinglab/NCS_data/expertdata_01_02_20_1206"
    # data_filename = "/home/graspinglab/NCS_data/Data_Box_S_01_03_20_2309"    
    # data_filename = "/home/graspinglab/NCS_data/Data_Box_M_01_05_20_1705"    
    # data_filename = "/home/graspinglab/NCS_data/Data_Box_B_01_06_20_1532"    
    # data_filename = "/home/graspinglab/NCS_data/Data_Cylinder_S_01_04_20_1701"    
    # data_filename = "/home/graspinglab/NCS_data/Data_Cylinder_M_01_06_20_0013"    
    # data_filename = "/home/graspinglab/NCS_data/Data_Cylinder_B_01_06_20_1922"    

    # num_epoch = 20
    # total_steps = 10000 # not used in the function
    # batch_size = 250
    # model_path = "/home/graspinglab/NCS_data/ExpertTrainedNet"
    # train_network(data_filename, 0.3, 5, total_steps, batch_size, model_path)

    num_epoch = 10000
    batch_size = 25
    files_dir = "/home/orochi/NCS_data/"
    #files = [files_dir + "Data_Box_B_01_16_20_1728", files_dir + "Data_Box_B_01_21_20_1951", files_dir + "Data_Box_M_01_16_20_1826", files_dir + "Data_Box_M_01_21_20_2005", files_dir + "Data_Box_S_01_16_20_2335", files_dir + "Data_Box_S_01_21_20_2106",files_dir + "Data_Cylinder_B_01_16_20_1405", files_dir + "Data_Cylinder_B_01_21_20_1956", files_dir + "Data_Cylinder_M_01_16_20_1505", files_dir + "Data_Cylinder_M_01_21_20_2012", files_dir + "Data_Cylinder_M_01_21_20_2012", files_dir + "Data_Cylinder_S_01_17_20_0009", files_dir + "Data_Cylinder_S_01_21_20_2118", files_dir + "Data_Hour_B_03_31_20_0348", files_dir + "Data_Hour_M_03_30_20_1756", files_dir + "Data_Hour_S_03_30_20_1743"]
    files = [files_dir + "Data_Hour_B_05_14_20_0750", files_dir + "Data_Hour_M_05_14_20_0652", files_dir +"Data_Hour_S_05_14_20_0557", files_dir +"Data_Cylinder_B_05_14_20_0453", files_dir +"Data_Cylinder_M_05_14_20_0407", files_dir +"Data_Cylinder_S_05_14_20_0319",files_dir +"Data_Box_B_05_14_20_0216", files_dir +"Data_Box_M_05_14_20_0131", files_dir +"Data_Box_S_05_14_20_0044"]
    all_training_set = []
    all_training_label = []
    all_testing_set = []
    all_testing_label = []
    normalize=False
    for i in range(len(files)):
        file = open(files[i] + ".pkl", "rb")
        data = pickle.load(file)
        file.close()
        state_input = np.array(data["states"])
        #print(np.shape(state_input))
        temp = state_input[:,0:49]
        state_input=np.append(temp,state_input[:,55:],axis=1)
        
        SI_size=np.shape(state_input)
        #print(data)
        grasp_label = np.array(data["grasp_success"])   
        '''
        state_input2=np.zeros([int(SI_size[0]/50),SI_size[1]])
        grasp_label2=np.zeros(int(SI_size[0]/50))
        for j in range(len(state_input)):
            if j %50 == 49:
                state_input2[int((i+1)/50)]=state_input[j]
                grasp_label2[int((i+1)/50)]=grasp_label[j]
        '''
        if normalize:
            ips=np.shape(state_input)
            for j in range(ips[1]):
                state_input[:,j]=normalize_vector(state_input[:,j])
        # extract training set
        num_inputs=np.shape(state_input)
        shape_type=np.zeros([num_inputs[0],1])
        #print(num_inputs)
        si=np.zeros([num_inputs[0],66])
        #print(np.shape(si))
        for j in range(num_inputs[0]):
            temp=state_input[j]
            #print(temp)
            si[j]=temp[0:66]
        #if i<6:
        #    shape_type[:]=1
        state_input=si
        #print(state_input)
        #state_input=np.append(state_input,shape_type,axis=1)
        #print(state_input)
        training_len = len(state_input) * 0.8
        if i == 0:
            testing_len= int(len(state_input) * 0.2)
        #CONVERT TO LOCAL
        #state_input=convert_to_local(state_input)
        
        training_set = state_input[0:int(training_len)]
        training_label = grasp_label[0:int(training_len)]

        # extract testing set
        testing_set = state_input[int(training_len):]
        testing_label = grasp_label[int(training_len):]
        
        all_training_set=all_training_set+list(training_set)
        all_testing_set=all_testing_set+list(testing_set)
        all_training_label=all_training_label+list(training_label)
        all_testing_label=all_testing_label+list(testing_label)
        print("label: ", i, len(training_set), len(training_label), len(testing_set), len(testing_label))
    # pdb.set_trace()
    all_training_set=np.array(all_training_set)
    all_testing_set=np.array(all_testing_set)
    all_training_label=np.array(all_training_label)
    all_testing_label=np.array(all_testing_label)
    print(np.shape(all_testing_set))
    #GP_net=trainGP(all_training_set,all_training_label,all_testing_set,all_testing_label)
    #classifier_net,total_percent,false_pos,true_pos=train_network(all_training_set, all_training_label, num_epoch, len(all_training_set), batch_size,all_testing_set,all_training_label)    
    model = ReducedLinearNetwork()
    model.load_state_dict(torch.load('trained_model_05_14_20_1349local.pt'))
    model.eval()
    print('model loaded')
    #output = 0.25
    
    total_percent=np.zeros(10)
    false_pos=np.zeros([10,6])
    true_pos=np.zeros([10,6])
    print(testing_len)
    for i in range(10):
        total_percent[i],true_pos[i,:],false_pos[i,:],false_pos_indicies,false_neg_indicies=test_network(all_testing_set, all_testing_label,model,0.1*i+0.0)    
        #p_show, f_show = get_false_grasps(false_pos_indicies,false_neg_indicies, all_testing_set[0:testing_len])
        #show_false_grasps(p_show,f_show,"Box","B")
    for j in range(1):
        print(true_pos[:,j],false_pos[:,j])
        plt.plot(false_pos[:,j],true_pos[:,j])
        plt.title('ROC Graph for most updated network')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.show()
