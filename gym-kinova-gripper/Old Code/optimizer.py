#!/usr/bin/env python3
import numpy as np
from classifier_network import LinearNetwork
import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Grasp_net = LinearNetwork().to(device)
trained_model = "/home/orochi/NCSGen-updated/NCSGen-master/gym-kinova-gripper/trained_model_01_23_20_0111.pt"
model = torch.load(trained_model)
Grasp_net.load_state_dict(model)
Grasp_net.eval()


def optimize_grasp(obs, init_class):
    # wrist positions
    initial_obs = obs
    initial_reward = init_class
    wrist_x = obs[18]
    wrist_y = obs[19]
    wrist_z = obs[20]
    iterations = 1000
    reward_list = {}
    best_reward = {}

    # take wrist position in obs. [18,19,20]
    # mutate that slightly
    # try it and get a new classifier result
    # store it for us to play with
    # vary x, vary y, vary z, vary together
    for i in range(4):
        new_obs = initial_obs
        best_reward[str(i)] = 0
        for k in range(iterations):
            if i == 3:
                # vary all of the positions
                new_x = random.uniform(-1, 1)
                new_y = random.uniform(-1, 1)
                new_z = random.uniform(-1, 1)
                new_obs[18] += new_x
                new_obs[19] += new_y
                new_obs[20] += new_z
                # feed into classifier
                network_inputs = new_obs[0:5]
                network_inputs = np.append(network_inputs, new_obs[6:23])
                network_inputs = np.append(network_inputs, new_obs[24:])
                inputs = torch.FloatTensor(np.array(network_inputs)).to(device)
                outputs = Grasp_net(inputs).cpu().data.numpy().flatten()
                
                if outputs[0] > best_reward[str(i)]:
                    obs = new_obs
                    best_reward[str(i)] = outputs[0]
                reward_list[str(i) + ' ' + str(k)] = outputs[0]

            else:
                # vary one of the positions
                new_value = random.uniform(-1, 1)
                new_obs[i+18] += new_value

                # feed into classifier
                network_inputs = new_obs[0:5]
                network_inputs = np.append(network_inputs, new_obs[6:23])
                network_inputs = np.append(network_inputs, new_obs[24:])
                inputs = torch.FloatTensor(np.array(network_inputs)).to(device)
                outputs = Grasp_net(inputs).cpu().data.numpy().flatten()

                if outputs[0] > best_reward[str(i)]:
                    obs = new_obs
                    best_reward[str(i)] = outputs[0]
                reward_list[str(i) + ' ' + str(k)] = outputs[0]
    return reward_list, best_reward

# obs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# print(optimize_grasp(obs,0))