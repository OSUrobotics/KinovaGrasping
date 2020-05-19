#!/usr/bin/env python3

"""
Author : Yi Herng Ong
Title : Script to collect data from robot at random joint configuration 
"""

# Set object at random close palm position
# Set random joint configuration for each object offsets
# Close fingers for 5 time steps, then lift
# Record whether it's success or fail

import gym
import numpy as np
import pdb
import time
import datetime
import pickle

def PID(target, current):
    err = target - current
    if err < 0.0:
        err = 0.0
    diff = err / 4
    vel = err + diff # PD control
    action = (vel / 0.8) * 0.3 
    
    return action

# Joint limits for starting configurations
# Small object -> 0.02: [0.4, 0.6], 0.03: [0.3,0.5], 0.04: [0.1, 0.3]
# Med object -> 0.02: [0.3, 0.5], 0.03: [0.2, 0.4], 0.04: [0.1, 0.2]
# Large object -> 0.02: [0.2, 0.4], 0.03: [0.1,0.3], 0.04: [0.0, 0.1]
def getRandomJoint(obj_size, obj_pose):
    # print(obj_pose)
    if obj_size == "S":
        # Two fingered side
        if obj_pose == -0.02:
            f2 = np.random.uniform(0.4, 0.61)
            f3 = np.random.uniform(0.4, 0.61)
            f1 = np.random.uniform(0.8, 1.1)
        # One fingered side
        if obj_pose == 0.02:
            f2 = np.random.uniform(0.8, 1.1)
            f3 = np.random.uniform(0.8, 1.1)
            f1 = np.random.uniform(0.4, 0.61) 
        if obj_pose == -0.03:
            f2 = np.random.uniform(0.3, 0.51)
            f3 = np.random.uniform(0.3, 0.51)
            f1 = np.random.uniform(0.8, 1.1)
        # One fingered side
        if obj_pose == 0.03:
            f2 = np.random.uniform(0.8, 1.1)
            f3 = np.random.uniform(0.8, 1.1)
            f1 = np.random.uniform(0.3, 0.51) 
        if obj_pose == -0.04:
            f2 = np.random.uniform(0.1, 0.31)
            f3 = np.random.uniform(0.1, 0.31)
            f1 = np.random.uniform(0.8, 1.1)
        # One fingered side
        if obj_pose == 0.04:
            f2 = np.random.uniform(0.8, 1.1)
            f3 = np.random.uniform(0.8, 1.1)
            f1 = np.random.uniform(0.1, 0.31)                         
        # Center
        elif abs(obj_pose - 0.0) < 0.001:
            f2 = np.random.uniform(0.6, 0.81)
            f3 = np.random.uniform(0.6, 0.81)
            f1 = np.random.uniform(0.6, 0.81)

    if obj_size == "M":
        # Two fingered side
        if obj_pose == -0.02:
            f2 = np.random.uniform(0.3, 0.51)
            f3 = np.random.uniform(0.3, 0.51)
            f1 = np.random.uniform(0.8, 1.1)
        # One fingered side
        if obj_pose == 0.02:
            f2 = np.random.uniform(0.8, 1.1)
            f3 = np.random.uniform(0.8, 1.1)
            f1 = np.random.uniform(0.3, 0.51) 
        if obj_pose == -0.03:
            f2 = np.random.uniform(0.2, 0.41)
            f3 = np.random.uniform(0.2, 0.41)
            f1 = np.random.uniform(0.8, 1.1)
        # One fingered side
        if obj_pose == 0.03:
            f2 = np.random.uniform(0.8, 1.1)
            f3 = np.random.uniform(0.8, 1.1)
            f1 = np.random.uniform(0.2, 0.41) 
        if obj_pose == -0.04:
            f2 = np.random.uniform(0.1, 0.21)
            f3 = np.random.uniform(0.1, 0.21)
            f1 = np.random.uniform(0.8, 1.1)
        # One fingered side
        if obj_pose == 0.04:
            f2 = np.random.uniform(0.8, 1.1)
            f3 = np.random.uniform(0.8, 1.1)
            f1 = np.random.uniform(0.1, 0.21)                         
        # Center
        elif abs(obj_pose - 0.0) < 0.001:
            f2 = np.random.uniform(0.5, 0.71)
            f3 = np.random.uniform(0.5, 0.71)
            f1 = np.random.uniform(0.5, 0.71)

    if obj_size == "B":
        # Two fingered side
        if obj_pose == -0.02:
            f2 = np.random.uniform(0.2, 0.41)
            f3 = np.random.uniform(0.2, 0.41)
            f1 = np.random.uniform(0.8, 1.1)
        # One fingered side
        if obj_pose == 0.02:
            f2 = np.random.uniform(0.8, 1.1)
            f3 = np.random.uniform(0.8, 1.1)
            f1 = np.random.uniform(0.2, 0.41) 
        if obj_pose == -0.03:
            f2 = np.random.uniform(0.1, 0.31)
            f3 = np.random.uniform(0.1, 0.31)
            f1 = np.random.uniform(0.8, 1.1)
        # One fingered side
        if obj_pose == 0.03:
            f2 = np.random.uniform(0.8, 1.1)
            f3 = np.random.uniform(0.8, 1.1)
            f1 = np.random.uniform(0.1, 0.31) 
        if obj_pose == -0.04:
            f2 = np.random.uniform(0.0, 0.11)
            f3 = np.random.uniform(0.0, 0.11)
            f1 = np.random.uniform(0.8, 1.1)
        # One fingered side
        if obj_pose == 0.04:
            f2 = np.random.uniform(0.8, 1.1)
            f3 = np.random.uniform(0.8, 1.1)
            f1 = np.random.uniform(0.0, 0.11)                         
        # Center
        elif abs(obj_pose - 0.0) < 0.001:
            f2 = np.random.uniform(0.4, 0.61)
            f3 = np.random.uniform(0.4, 0.61)
            f1 = np.random.uniform(0.4, 0.61)

    return np.array([f1, f2, f3])

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

def DataCollection_GraspClassifier(episode_num, obj_shape, obj_size, save=True):
    env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    start = time.time()
    graspSuccess_label = []
    obs_label = []
    print('AWOOOOOOOOOGA')
    for episode in range(episode_num):
        obs, done = env.reset(), False
        reward = 0
        target_joint_config = getRandomJoint(obj_size, obs[21])
        step = 0
        reach = False
        episode_obs_label = []
        random_finger_action = getRandomVelocity()
        while not done:
            # finger action 
            finger_action = []
            # First part : go to random joint config         
            # print((np.array(obs[25:28])))
            if (np.max(np.abs((np.array(obs[25:28]) - target_joint_config))) > 0.01) and (reach == False):
                for finger in range(3):
                    # print(target_joint_config[i], obs[25+i])
                    finger_action.append(PID(target_joint_config[finger], obs[25+finger]))
                action = np.array([0.0, finger_action[0], finger_action[1], finger_action[2]])
                episode_obs_label=obs
            # Second part : close fingers
            else:
                reach = True # for not going back to the previous if loop
                step += 1     
                if step >= 50 and step < 100: # wait for one second
                    episode_obs_label=obs # collect observation data after reach random joint config
                    #print(episode_obs_label)
                    # action = np.array([0.0, 0.2, 0.2, 0.2])
                    action = np.array([0.0, random_finger_action[0], random_finger_action[1], random_finger_action[2]])
                elif step > 100:
                    # finger_action = getRandomVelocity()
                    action = np.array([0.3, random_finger_action[0], random_finger_action[1], random_finger_action[2]])
                else:
                    action = np.array([0.0, 0.0, 0.0, 0.0])

            # print(step)
            obs, reward, done, _ = env.step(action)
            #print(done)
            # env.render()

        # If object is lifted,     

        if reward:
            graspSuccess_label.append(1)
        else:
            graspSuccess_label.append(0)
        obs_label.append(episode_obs_label)
        #print(obs_label)
        print(episode)
    # print(time.time() - start)
    # pdb.set_trace()

    if save:
        filename = "/home/orochi/NCS_data/Data" + "_{}".format(obj_shape) + "_{}".format(obj_size)
        print("Saving...")
        data = {}
        data["states"] = obs_label
        data["grasp_success"] = graspSuccess_label
        # data["action"] = action_label
        # data["total_steps"] = total_steps
        file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
        pickle.dump(data, file)
        file.close()    



# DataCollection_GraspClassifier(5000, "Box", "S", True)
# DataCollection_GraspClassifier(5000, "Box", "M", True)
# DataCollection_GraspClassifier(5000, "Box", "B", True)
# DataCollection_GraspClassifier(5000, "Cylinder", "S", True)
# DataCollection_GraspClassifier(5000, "Cylinder", "M", True)
# DataCollection_GraspClassifier(5000, "Cylinder", "B", True)


