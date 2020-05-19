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


def roughEquals(x,y):
    if (x > y - 0.001) & (x < y +0.001):
        return True
    else:
        return False
# Joint limits for starting configurations
# Small object -> 0.02: [0.4, 0.6], 0.03: [0.3,0.5], 0.04: [0.1, 0.3]
# Med object -> 0.02: [0.3, 0.5], 0.03: [0.2, 0.4], 0.04: [0.1, 0.2]
# Large object -> 0.02: [0.2, 0.4], 0.03: [0.1,0.3], 0.04: [0.0, 0.1]
def getRandomJoint(obj_size, obj_pose, obj_shape):
    # print(obj_pose)    
    #print(obj_size,obj_pose)
    if obj_size == "S":
        # Two fingered side
        if (obj_pose > -0.02) and (obj_pose <= 0):
            f2 = np.random.uniform(0.4, 0.65)
            f3 = np.random.uniform(0.4, 0.65)
            f1 = np.random.uniform(0.4, 1.15)
            #print('number 1')
        # One fingered side
        if (obj_pose < 0.02) and (obj_pose >= 0):
            f2 = np.random.uniform(0.4, 1.15)
            f3 = np.random.uniform(0.4, 1.15)
            f1 = np.random.uniform(0.4, 0.65) 
            #print('number 2')
        if (obj_pose > -0.03) and (obj_pose <= -0.02):
            f2 = np.random.uniform(0.4, 0.55)
            f3 = np.random.uniform(0.4, 0.55)
            f1 = np.random.uniform(0.4, 1.15)
            #print('number 3')
        # One fingered side
        if (obj_pose < 0.03) and (obj_pose >= 0.02):
            f2 = np.random.uniform(0.4, 1.15)
            f3 = np.random.uniform(0.4, 1.15)
            f1 = np.random.uniform(0.4, 0.55) 
            #print('number 4')
        if (obj_pose > -0.04) and (obj_pose <= -0.03):
            f2 = np.random.uniform(0.2, 0.35)
            f3 = np.random.uniform(0.2, 0.35)
            f1 = np.random.uniform(0.2, 1.15)
            #print('number 5')
        # One fingered side
        if (obj_pose < 0.04) and (obj_pose >= 0.03):
            f2 = np.random.uniform(0.2, 1.15)
            f3 = np.random.uniform(0.2, 1.15)
            f1 = np.random.uniform(0.2, 0.35) 
            #print('number 6')
            
        if (obj_pose <-0.04):
            f2 = np.random.uniform(0.1, 0.25)
            f3 = np.random.uniform(0.1, 0.25)
            f1 = np.random.uniform(0.1, 1.15)
            #print('number 7')
        # One fingered side.1
        if (obj_pose > 0.04):
            f2 = np.random.uniform(0.1, 1.15)
            f3 = np.random.uniform(0.1, 1.15)
            f1 = np.random.uniform(0.1, 0.25)  
            #print('number 8')                     
        # Center
        elif abs(obj_pose - 0.0) < 0.001:
            f2 = np.random.uniform(0, 1.0)
            f3 = np.random.uniform(0, 1.0)
            f1 = np.random.uniform(0, 1.0)
            #print('number 9')

    if obj_size == "M":
        # Two fingered side
        if (obj_pose > -0.02) and (obj_pose <= 0):
            f2 = np.random.uniform(0, 0.51)
            f3 = np.random.uniform(0, 0.51)
            f1 = np.random.uniform(0, 1.1)
        # One fingered side
        if (obj_pose < 0.02) and (obj_pose >= 0):
            f2 = np.random.uniform(0, 1.1)
            f3 = np.random.uniform(0, 1.1)
            f1 = np.random.uniform(0, 0.51) 
        if (obj_pose > -0.03) and (obj_pose <= -0.02):
            f2 = np.random.uniform(0, 0.41)
            f3 = np.random.uniform(0, 0.41)
            f1 = np.random.uniform(0, 1.1)
        # One fingered side
        if (obj_pose < 0.03) and (obj_pose >= 0.02):
            f2 = np.random.uniform(0, 1.1)
            f3 = np.random.uniform(0, 1.1)
            f1 = np.random.uniform(0, 0.41) 
        if (obj_pose < -0.03):
            f2 = np.random.uniform(0, 0.21)
            f3 = np.random.uniform(0, 0.21)
            f1 = np.random.uniform(0, 1.1)
        # One fingered side
        if (obj_pose > 0.03):
            f2 = np.random.uniform(0, 1.1)
            f3 = np.random.uniform(0, 1.1)
            f1 = np.random.uniform(0, 0.21)                         
        # Center
        elif abs(obj_pose - 0.0) < 0.001:
            f2 = np.random.uniform(0, 0.71)
            f3 = np.random.uniform(0, 0.71)
            f1 = np.random.uniform(0, 0.71)

    if obj_size == "B":
        # Two fingered side
        if (obj_pose > -0.02) and (obj_pose <= 0):
            f2 = np.random.uniform(0, 0.41)
            f3 = np.random.uniform(0, 0.41)
            f1 = np.random.uniform(0, 1.1)
        # One fingered side
        if (obj_pose < 0.02) and (obj_pose >= 0):
            f2 = np.random.uniform(0, 1.1)
            f3 = np.random.uniform(0, 1.1)
            f1 = np.random.uniform(0, 0.41) 
        if (obj_pose > -0.03) and (obj_pose <= -0.02):
            f2 = np.random.uniform(0, 0.31)
            f3 = np.random.uniform(0, 0.31)
            f1 = np.random.uniform(0, 1.1)
        # One fingered side
        if (obj_pose < 0.03) and (obj_pose >= 0.02):
            f2 = np.random.uniform(0, 1.1)
            f3 = np.random.uniform(0, 1.1)
            f1 = np.random.uniform(0, 0.31) 
        if (obj_pose < -0.03):
            f2 = np.random.uniform(0.0, 0.11)
            f3 = np.random.uniform(0.0, 0.11)
            f1 = np.random.uniform(0, 1.1)
        # One fingered side
        if (obj_pose > 0.03):
            f2 = np.random.uniform(0, 1.1)
            f3 = np.random.uniform(0, 1.1)
            f1 = np.random.uniform(0.0, 0.11)                         
        # Center
        elif abs(obj_pose - 0.0) < 0.001:
            f2 = np.random.uniform(0, 0.61)
            f3 = np.random.uniform(0, 0.61)
            f1 = np.random.uniform(0, 0.61)

    return np.array([f1, f2, f3])

def getRandomVelocity():
    # if not flag:
    flag = np.random.choice(np.array([1, 0]), p = [0.8, 0.2])
    if flag:
        f1 = np.random.uniform(0.0, 0.3)
        f2 = np.random.uniform(0.0, 0.3)
        f3 = np.random.uniform(0.0, 0.3)
        #print('flag')
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
        obs, done = env.reset(start_pos=None, obj_params=[obj_shape,obj_size]), False
        #print(obs[21:24])
        reward = 0
        target_joint_config = getRandomJoint(obj_size, obs[21], obj_shape)
        #print(target_joint_config)
        #print(obs[25:28])
        step = 0
        prelim_step=0
        reach = False
        episode_obs_label = []
        random_finger_action = getRandomVelocity()
        while not done:
            # finger action 
            finger_action = []
            # First part : go to random joint config         
            # print((np.array(obs[25:28])))
            if (np.max(np.abs((np.array(obs[25:28]) - target_joint_config))) > 0.1) and (reach == False) and prelim_step <200:
                for finger in range(3):
                    # print(target_joint_config[i], obs[25+i])
                    finger_action.append(PID(target_joint_config[finger], obs[25+finger]))
                action = np.array([0.0, 0.0, 0.0, finger_action[0], finger_action[1], finger_action[2]])
                episode_obs_label=obs
                prelim_step+=1
            # Second part : close fingers
            else:
                reach = True # for not going back to the previous if loop
                step += 1     
                if step >= 25 and step < 75: # wait for one second
                    if step >70:
                        episode_obs_label=obs # collect observation data after reach random joint config
                    #print('closing')
                    # action = np.array([0.0, 0.2, 0.2, 0.2])
                    action = np.array([0.0, 0.0, 0.0, random_finger_action[0], random_finger_action[1], random_finger_action[2]])
                elif step > 75:
                    # finger_action = getRandomVelocity()
                    action = np.array([0.0, 0.0, 0.15, 0.05, 0.05, 0.05])
                    #print('lifting')
                else:
                    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            #print(step)
            #print('data collection pre', done)
            obs, reward, done, _ = env.step(action)
            #print(obs[-1])
            #print('data collection', done)
            #env.render()

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
        print(np.average(graspSuccess_label))
    # print(time.time() - start)
    # pdb.set_trace()

    if save:
        filename = "/home/orochi/NCS_data/Data" + "_{}".format(obj_shape) + "_{}".format(obj_size) + "LOCAL"
        print("Saving...")
        data = {}
        print(np.average(graspSuccess_label))
        data["states"] = obs_label
        data["grasp_success"] = graspSuccess_label
        # data["action"] = action_label
        # data["total_steps"] = total_steps
        file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
        pickle.dump(data, file)
        file.close()    

DataCollection_GraspClassifier(1000, "Box", "S", True)
DataCollection_GraspClassifier(1000, "Box", "M", True)
DataCollection_GraspClassifier(1000, "Box", "B", True)
       
DataCollection_GraspClassifier(1000, "Cylinder", "S", True)
DataCollection_GraspClassifier(1000, "Cylinder", "M", True)
DataCollection_GraspClassifier(1000, "Cylinder", "B", True)
        
DataCollection_GraspClassifier(1000, "Hour", "S", True)
DataCollection_GraspClassifier(1000, "Hour", "M", True)
DataCollection_GraspClassifier(1000, "Hour", "B", True)

#DataCollection_GraspClassifier(100, "Box", "S", True)
#DataCollection_GraspClassifier(100, "Hour", "M", True)
#DataCollection_GraspClassifier(100, "Hour", "B", True)


