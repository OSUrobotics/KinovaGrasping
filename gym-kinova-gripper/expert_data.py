#!/usr/bin/env python3

'''
Author : Yi Herng Ong
Date : 9/29/2019
Purpose : Supervised learning for near contact grasping strategy
'''

import os, sys
import numpy as np
import gym
from gym import  wrappers # Used for video rendering
import argparse
import pdb
import pickle
import datetime
#from NCS_nn import NCS_net, GraspValid_net
import torch 
from copy import deepcopy
# from gen_new_env import gen_new_obj
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################################################################

###########################################
# Previous PID Nudge Controller w/ 0.8 rad/s joint velocity
###########################################
class expert_PID():
    def __init__(self, action_space):
        self.kp = 1
        self.kd = 1
        self.ki = 1
        self.prev_err = 0.0
        self.sampling_time = 0.04
        self.action_range = [action_space.low, action_space.high]
        self.x = 0.0
        self.flag = 1
        # print(action_space.high)
    # expert vel when object moves
    def get_PID_vel(self, dot_prod):
        err = 1 - dot_prod 
        diff = (err) / self.sampling_time
        vel = err * self.kp + diff * self.kd
        self.prev_err = err
        return vel

    def map_action(self, vel):
        return self.action_range[1][-1]*(vel / 13.637)

    def get_expert_vel(self, dot_prod, dominant_finger):
        vel = self.get_PID_vel(dot_prod)
        clipped_vel = self.map_action(vel)
        # pdb.set_trace()
        wrist = 0.0
        if dominant_finger > 0:
            f1 = clipped_vel
            f2 = clipped_vel * 0.8 
            f3 = clipped_vel * 0.8
        else:
            f1 = clipped_vel * 0.8
            f2 = clipped_vel 
            f3 = clipped_vel 
        return np.array([wrist, f1, f2, f3])

    def get_expert_move_to_touch(self, dot_prod, dominant_finger):
        max_move_vel = self.get_PID_vel(dot_prod)
        clipped_vel = self.map_action(max_move_vel)
        self.x += (self.action_range[1][-1] - clipped_vel) * 0.04
        if self.x >= clipped_vel: # if x accumulate to clipped vel, use clipped vel instead
            self.x = clipped_vel
        if self.x < 0.0:
            self.x = 0.0
        vel = 0.8 - self.x
        # pdb.set_trace()
        # if dominant_finger > 0:
        #     f1 = vel
        #     f2 = vel * 0.8
        #     f3 = vel * 0.8
        # else:
        f1 = vel
        f2 = vel 
        f3 = vel 
        return np.array([0.0, f1, f2, f3])

    def generate_expert_move_to_close(self, vel, max_vel, dominant_finger):
        # if self.flag == 1:
        #     # print("only here once")
        self.vel = vel[:]
        # self.flag = 0
        wrist = 0.0
        # f_vel = self.vel[:]
        for i in range(3):
            if self.vel[i+1] < max_vel:
                # print("here")
                self.vel[i+1] += 0.1*self.vel[i+1]
            else:
                self.vel[i+1] = max_vel
        return np.array([wrist, self.vel[1], self.vel[2]*0.7, self.vel[3]*0.7])


def generate_lifting_data(env, total_steps, filename, grasp_filename):
    states = []
    label = []
    # file = open(filename + ".pkl", "rb")
    # data = pickle.load(file)
    # file.close()
    import time
    print(total_steps)
    for step in range(int(total_steps)):
        _, curr_actions, reward = env.reset()
        if reward == 1:
            continue
        else:    
            for i in range(40):
                # lift
                if i < 20:
                    action = np.array([0.0, 0.0, 0.0, 0.0])
                else:
                    action = np.array([0.2, curr_actions[1], curr_actions[2], curr_actions[3]])
                # pdb.set_trace()
                obs, reward , _, _ = env.step(action)
        
            # time.sleep(0.25)    
        print(reward)

        # record fail or success
        label.append(reward)
        if step % 10000 == 9999:
            print("Steps:{}".format(step + 1))

    print("Finish collecting grasp validation data, saving...")
    grasp_data = {}
    grasp_data["grasp_sucess"] = label
    grasp_data["states"] = data["states"][0:int(total_steps)]
    grasp_file = open(grasp_filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
    pickle.dump(grasp_data, grasp_file)
    grasp_file.close()


# function that generates expert data for grasping an object from ground up
def generate_Data(env, num_episode, filename, replay_buffer):
    # grasp_net = GraspValid_net(35).to(device)
    # trained_model = "data_cube_7_grasp_classifier_10_16_19_1509.pt"

    # model = torch.load(trained_model)
    # grasp_net.load_state_dict(model)
    # grasp_net.eval()
    env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    # preaction
    wrist = 0.0
    f1 = 0.8
    f2 = f1 * 0.8
    f3 = f1 * 0.8

    # postaction
    wrist_post = 0.0
    f1_post = 0.65
    f2_post = 0.55
    f3_post = 0.55
    
    #lift
    wrist_lift = 0.2
    f1_lift = 0.4
    f2_lift = 0.4
    f3_lift = 0.4

    move_to_touch_action = np.array([wrist, f1, f2, f3])
    move_to_close_action = np.array([wrist_post, f1_post, f2_post, f3_post])
    move_to_lift = np.array([wrist_lift, f1_lift, f2_lift, f3_lift])

    expert = expert_PID(env.action_space)
    episode_timesteps = 0

    # display filename
    # print("filename:", filename)

    label = []
    states = []
    states_for_lifting = []
    label_for_lifting = []
    states_when_closing = []
    label_when_closing = []
    states_ready_grasp = []
    label_ready_grasp = []
    states_all_episode = []
    obs_all_episode = []
    action_all_episode = []
    nextobs_all_episode = []
    reward_all_episode = []
    done_all_episode = []

    for episode in range(num_episode):
        # env = gym.make('gym_kinova_gripper:kinovagripper-v0')
        obs, done = env.reset(), False
        dom_finger = env.env._get_obj_pose()[0] # obj's position in x
        ini_dot_prod = env.env._get_dot_product(env.env._get_obj_pose()) # 
        action = expert.get_expert_move_to_touch(ini_dot_prod, dom_finger)
        touch = 0
        not_close = 1
        close = 0
        touch_dot_prod = 0.0
        t = 0    
        cum_reward = 0.0
        states_each_episode = []
        obs_each_episode = []
        action_each_episode = []
        nextobs_each_episode = []
        reward_each_episode = []
        done_each_episode = []

        for _ in range(100):
            states.append(obs)
            label.append(action)    
            states_each_episode.append(obs)
            
            obs_each_episode.append(obs)

            # pdb.set_trace()
            next_obs, reward, done, _ = env.step(action)
            jA_action = next_obs[24:28][:]
            jA_action[0] = (jA_action[0] / 0.2) * 1.5
            action_each_episode.append(jA_action) # get joint angle as action
            nextobs_each_episode.append(next_obs)
            reward_each_episode.append(reward)
            done_each_episode.append(done)

            # env.render()
            # print(action)
            # store data into replay buffer 
            replay_buffer.add(obs, action, next_obs, reward, done)
            # replay_buffer.add(obs, obs[24:28], next_obs, reward, done) # store joint angles as actions

            cum_reward += reward
            # print(next_obs[0:7])
            obs = next_obs
            # dot_prod = obs[-1]
            dot_prod = env.env._get_dot_product(env.env._get_obj_pose())            
            # move closer towards object
            if touch == 0:
                action = expert.get_expert_move_to_touch(dot_prod, dom_finger)
            # contact with object, PID control with object pose as feedback
            if abs(dot_prod - ini_dot_prod) > 0.001 and not_close == 1:                
                action = expert.get_expert_vel(dot_prod, dom_finger)
                prev_vel = action
                touch_dot_prod = dot_prod
            # if object is close to center 
            if touch_dot_prod > 0.8: # can only check dot product after fingers are making contact
                close = 1                
            if close == 1:
                action = expert.generate_expert_move_to_close(prev_vel, 0.6, dom_finger)
                not_close = 0
                # when it's time to lift
                states_for_lifting.append(obs)
                label_for_lifting.append(action)
                if t > 60:
                    action[0] = 0.8
            if t > 50:
                states_ready_grasp.append(obs)
                label_ready_grasp.append(action)    
            if t <= 50:
                states_when_closing.append(obs)
                label_when_closing.append(action)
            t += 1
            # print(next_obs[24:31])
        # pdb.set_trace()
        states_all_episode.append(states_each_episode)
        obs_all_episode.append(obs_each_episode)
        action_all_episode.append(action_each_episode)
        nextobs_all_episode.append(nextobs_each_episode)
        reward_all_episode.append(reward_each_episode)
        done_all_episode.append(done_each_episode)

        # print("Collecting.., num_episode:{}".format(episode))
        # pdb.set_trace()
    # print("saving...")
    # data = {}
    # data["states_all_episode"] = states_all_episode
    # pdb.set_trace()

    # data["states"] = states
    # data["label"] = label
    # data["states_for_lifting"] = states_for_lifting
    # data["label_for_lifting"] = label_for_lifting
    # data["states_when_closing"] = states_when_closing
    # data["label_when_closing"] = label_when_closing    
    # data["states_ready_grasp"] = states_ready_grasp
    # data["label_ready_grasp"] = label_ready_grasp

    ### Data collection for joint angle action space ###
    # data["obs"] = obs_all_episode
    # data["action"] = action_all_episode
    # data["next_obs"] = nextobs_all_episode
    # data["reward"] = reward_all_episode
    # data["done"] = done_all_episode
    # file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
    # pickle.dump(data, file)
    # file.close()
    # return data

    return replay_buffer

#################################################################################################



class PID(object):
    def __init__(self, action_space):
        self.kp = 1
        self.kd = 1
        self.ki = 1
        self.prev_err = 0.0
        self.sampling_time = 4
        self.action_range = [action_space.low, action_space.high]

    def velocity(self, dot_prod):
        err = 1 - dot_prod
        diff = (err) / self.sampling_time
        vel = err * self.kp + diff * self.kd
        action = (vel / 1.25) * 0.3 # 1.25 means dot product equals to 1
        if action < 0.05:
            action = 0.05
        return action

    def joint(self, dot_prod):
        err = 1 - dot_prod 
        diff = (err) / self.sampling_time
        joint = err * self.kp + diff * self.kd
        action = (joint / 1.25) * 2 # 1.25 means dot product equals to 1
        return action

    def touch_vel(self, obj_dotprod, finger_dotprod):
        err = obj_dotprod - finger_dotprod
        diff = err / self.sampling_time
        vel = err * self.kp + diff * self.kd
        action = (vel) * 0.3
        #if action < 0.8:
        #    action = 0.8
        if action < 0.05: # Old velocity
            action = 0.05
        #print("TOUCH VEL, action: ",action)
        return action

    def check_grasp(self, f_dist_old, f_dist_new):
        """
        Uses the current change in x,y position of the distal finger tips, summed over all fingers to determine if
        the object is grasped (fingers must have only changed in position over a tiny amount to be considered done).
        @param: f_dist_old: Distal finger tip x,y,z coordinate values from previous timestep
        @param: f_dist_new: Distal finger tip x,y,z coordinate values from current timestep
        """

        # Initial check to see if previous state has been set
        if f_dist_old is None:
            return False

        # Change in finger 1 distal x-coordinate position
        f1_change = abs(f_dist_old[0] - f_dist_new[0])
        f1_diff = f1_change / self.sampling_time
        #print("f1_change: ",f1_change)
        #print("f1_diff: ", f1_diff)
        #print("self.sampling_time: ",self.sampling_time)

        # Change in finger 2 distal x-coordinate position
        f2_change = abs(f_dist_old[3] - f_dist_new[3])
        f2_diff = f2_change / self.sampling_time

        # Change in finger 3 distal x-coordinate position
        f3_change = abs(f_dist_old[6] - f_dist_new[6])
        f3_diff = f3_change / self.sampling_time

        # Sum of changes in distal fingers
        f_all_change = f1_diff + f2_diff + f3_diff

        #print("f_all_change: ",f_all_change)

        # If the fingers have only changed a small amount, we assume the object is grasped
        if f_all_change < 0.00005:
            #print("RETURN TRUE")
            return True
        else:
            #print("***************************f_all_change: ", f_all_change)
            #print("RETURN FALSE")
            return False


##############################################################################
### PID nudge controller ###
# 1. Obtain (noisy) initial position of an object
# 2. move fingers that closer to the object
# 3. Move the other when the object is almost at the center of the hand 
# 4. Close grasp

### PID nudge controller ###
# 1. Obtain (noisy) initial position of an object
# 2. Move fingers that further away to the object 
# 3. Close the other finger (nearer one) and make contact "simultaneously"
# 4. Close fingers to secure grasp
##############################################################################
class ExpertPIDController(object):
    def __init__(self, states):
        self.prev_f1jA = 0.0
        self.prev_f2jA = 0.0
        self.prev_f3jA = 0.0
        self.step = 0.0
        self.init_obj_pose = states[21]  # X position of object
        #self.init_obj_pose = self._sim.data.get_geom_xpos(object)
        self.init_dot_prod = states[81]  # dot product of object wrt palm
        self.f1_vels = []
        self.f2_vels = []
        self.f3_vels = []
        self.wrist_vels = []

    def _count(self):
        self.step += 1

    def NudgeController(self, prev_states, states, action_space, label):
        # Define pid controller
        pid = PID(action_space)

        # obtain robot and object state
        robot_pose = np.array([states[0], states[1], states[2]])
        #obj_pose = states[21]  # X position of object
        obj_dot_prod = states[81]  # dot product of object wrt palm
        f1_jA = states[25]
        f2_jA = states[26]
        f3_jA = states[27]
        # Define target region
        x = 0.0
        y = -0.01
        target_region = [x, y]
        max_vel = 0.3
        # Define finger action
        f1 = 0.0  # far out finger, on single side
        f2 = 0.0  # double side finger - right of finger 1
        f3 = 0.0  # double side finger - left of finger 1
        wrist = 0.0

        # Distal finger x,y,z positions f1_dist, f2_dist, f3_dist
        if prev_states is None: # None if on the first timestep
            f_dist_old = None
        else:
            f_dist_old = prev_states[9:17]
        f_dist_new = states[9:17]

        #print("________________________")
        #print("f_dist_old: ",f_dist_old)
        #print("f_dist_new: ", f_dist_new)
        #print("________________________")

        #print("***NUDGE CONTROLLER self.init_obj_pos: ", self.init_obj_pose)
        ready_for_lift = False

        # Check if the object is near the center area (less than x-axis 0.03)
        if abs(self.init_obj_pose) <= 0.03:
            # only comparing initial X position of object. because we know
            # the hand starts at the same position every time (close to origin)
            """
            note that the object is near the center, so we just kinda close the fingers here
            """
            #print("CHECK 1: Object is near the center")
            f1, f2, f3 = 0.2, 0.2, 0.2 # Old velocity: 0.2, 0.2, 0.2  # set constant velocity of all three fingers

            # Check if change in object dot product to wrist center versus the initial dot product is greater than 0.01
            if abs(obj_dot_prod - self.init_dot_prod) > 0.01:  # start lowering velocity of the three fingers
                #print("CHECK 2: Obj dot product to wrist has changed more than 0.01")
                f1, f2, f3 = 0.2, 0.1, 0.1  # slows down to keep steady in one spot

                # Check if object is grasped - distal finger distance hasn't moved
                if pid.check_grasp(f_dist_old, f_dist_new) is True:
                    #print("Check 2A: Object is grasped, now lifting")
                    #wrist, f1, f2, f3 = 0.6, 0.15, 0.3, 0.3
                    #f1, f2, f3 = 0.3, 0.15, 0.15
                    #wrist = 0.3
                    ready_for_lift = True
        else:
            #print("CHECK 3: Object is on extreme left OR right sides")
            # object on right hand side, move 2-fingered side
            # Local representation: POS X --> object is on the RIGHT (two fingered) side of hand
            if self.init_obj_pose > 0.0:
                #print("CHECK 4: Object is on RIGHT side")
                # Pre-contact
                # Only Small change in object dot prod to wrist from initial position, must move more
                if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
                    #print("CHECK 5: Only Small change in object dot prod to wrist, moving f2 & f3")
                    f2 = pid.touch_vel(obj_dot_prod, states[79])  # f2_dist dot product to object
                    f3 = f2  # other double side finger moves at same speed
                    f1 = 0.0  # frontal finger doesn't move
                    wrist = 0.0
                # Post-contact
                else:  # now finger-object distance has been changed a decent amount.
                    #print("CHECK 6: Object dot prod to wrist has Changed More than 0.01")
                    # Goal is 1 b/c obj_dot_prod is based on comparison of two normalized vectors
                    if abs(1 - obj_dot_prod) > 0.01:
                        #print("CHECK 7: Obj dot prod to wrist is > 0.01, so moving ALL f1, f2 & f3")
                        # start to close the PID stuff
                        f2 = pid.velocity(obj_dot_prod)  # get PID velocity
                        f3 = f2  # other double side finger moves at same speed
                        f1 = 0.05 # Old velocity: 0.05  # frontal finger moves slightly
                        wrist = 0.0
                    else:  # goal is within 0.01 of being reached:
                        #print("CHECK 8: Obj dot prod to wrist is Within reach of 0.01 or less, Move F1 Only")
                        # start to close from the first finger
                        f1 = pid.touch_vel(obj_dot_prod, states[78])  # f1_dist dot product to object
                        f2 = 0.0
                        f3 = 0.0
                        wrist = 0.0

                    #print("Check 9a: Check for grasp (small distal finger movement)")
                    # Check for a good grasp (small distal finger movement)
                    if pid.check_grasp(f_dist_old, f_dist_new) is True:
                        #print("CHECK 9: Yes! Good grasp, move ALL fingers")
                        #wrist, f1, f2, f3 = 0.6, 0.15, 0.3, 0.3
                        #f1, f2, f3 = 0.3, 0.15, 0.15
                        #wrist = 0.3
                        ready_for_lift = True

            # object on left hand side, move 1-fingered side
            # Local representation: NEG X --> object is on the LEFT (thumb) side of hand
            else:
                #print("CHECK 10: Object is on the LEFT side")
                # Pre-contact
                # Only Small change in object dot prod to wrist from initial position, must move more
                if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
                    #print("CHECK 11: Only Small change in object dot prod to wrist, moving F1")
                    f1 = pid.touch_vel(obj_dot_prod, states[78])  # f1_dist dot product to object
                    f2 = 0.0
                    f3 = 0.0
                    wrist = 0.0
                # Post-contact
                else:  # now finger-object distance has been changed a decent amount.
                    #print("CHECK 12: Object dot prod to wrist has Changed More than 0.01")
                    # Goal is 1 b/c obj_dot_prod is based on comparison of two normalized vectors
                    if abs(1 - obj_dot_prod) > 0.01:
                        #print("CHECK 13: Obj dot prod to wrist is > 0.01, so kep moving f1, f2 & f3")
                        f1 = pid.velocity(obj_dot_prod)
                        f2 = 0.05
                        f3 = 0.05
                        wrist = 0.0
                    else:
                        # Goal is within 0.01 of being reached:
                        #print("CHECK 14: Obj dot prod to wrist is Within reach of 0.01 or less, Move F2 & F3 Only")
                        # start to close from the first finger
                        # nudge with thumb
                        f2 = pid.touch_vel(obj_dot_prod, states[79]) # f2_dist dot product to object
                        f3 = f2
                        f1 = 0.0
                        wrist = 0.0

                    #print("Check 15a: Check for grasp (small distal finger movement)")
                    # Check for a good grasp (small distal finger movement)
                    if pid.check_grasp(f_dist_old, f_dist_new) is True:
                        #print("CHECK 15: Good grasp - moving ALL fingers")
                        wrist, f1, f2, f3 = 0.6, 0.15, 0.3, 0.3
                        #f1, f2, f3 = 0.3, 0.15, 0.15
                        #wrist = 0.3
                        ready_for_lift = True

        #if self.step <= 400: # Old timesteps 400:
        #    label.append(0)
        #else:
        #    label.append(1)
        self._count()
        # print(self.step)

        #f1 *= 2
        #f2 *= 2
        #f3 *= 2

        f1 *= 3
        f2 *= 3
        f3 *= 3

        #print("f1: ",f1," f2: ",f2," f3: ",f3," wrist: ",wrist)
        self.f1_vels.append(f1)
        self.f2_vels.append(f2)
        self.f3_vels.append(f3)
        self.wrist_vels.append(f3)

        return np.array([wrist, f1, f2, f3]), label, ready_for_lift, self.f1_vels, self.f2_vels, self.f3_vels, self.wrist_vels  # action, grasp label, ready for lift

def GenerateTestPID_JointVel(obs,env):
    grasp_label=[]
    controller = ExpertPIDController(obs)
    action,grasp_label=controller.NudgeController(obs, env.action_space,grasp_label)
    return action

def save_coordinates(x,y,filename):
    np.save(filename+"_x_arr", x)
    np.save(filename+"_y_arr", y)

def add_heatmap_coords(expert_success_x,expert_success_y,expert_fail_x,expert_fail_y,obj_coords,info):
    if (info["lift_reward"] > 0):
        #print("add_heatmap_coords, lift_success TRUE")
        lift_success = True
    else:
        lift_success = False
        #print("add_heatmap_coords, lift_success FALSE")

    # Heatmap postion data - get starting object position and mark success/fail based on lift reward
    if (lift_success):
        # Get object coordinates, transform to array
        x_val = obj_coords[0]
        y_val = obj_coords[1]
        x_val = np.asarray(x_val).reshape(1)
        y_val = np.asarray(y_val).reshape(1)

        # Append initial object coordinates to Successful coordinates array
        expert_success_x = np.append(expert_success_x, x_val)
        expert_success_y = np.append(expert_success_y, y_val)

    else:
        # Get object coordinates, transform to array
        x_val = obj_coords[0]
        y_val = obj_coords[1]
        x_val = np.asarray(x_val).reshape(1)
        y_val = np.asarray(y_val).reshape(1)

        # Append initial object coordinates to Failed coordinates array
        expert_fail_x = np.append(expert_fail_x, x_val)
        expert_fail_y = np.append(expert_fail_y, y_val)

    ret = [expert_success_x, expert_success_y, expert_fail_x, expert_fail_y]
    return ret

def naive_check_grasp(f_dist_old, f_dist_new):
    """
    Uses the current change in x,y position of the distal finger tips, summed over all fingers to determine if
    the object is grasped (fingers must have only changed in position over a tiny amount to be considered done).
    @param: f_dist_old: Distal finger tip x,y,z coordinate values from previous timestep
    @param: f_dist_new: Distal finger tip x,y,z coordinate values from current timestep
    """

    # Initial check to see if previous state has been set
    if f_dist_old is None:
        return False
    sampling_time = 4

    # Change in finger 1 distal x-coordinate position
    f1_change = abs(f_dist_old[0] - f_dist_new[0])
    f1_diff = f1_change / sampling_time
    #print("f1_change: ",f1_change)
    #print("f1_diff: ", f1_diff)
    #print("self.sampling_time: ",self.sampling_time)

    # Change in finger 2 distal x-coordinate position
    f2_change = abs(f_dist_old[3] - f_dist_new[3])
    f2_diff = f2_change / sampling_time

    # Change in finger 3 distal x-coordinate position
    f3_change = abs(f_dist_old[6] - f_dist_new[6])
    f3_diff = f3_change / sampling_time

    # Sum of changes in distal fingers
    f_all_change = f1_diff + f2_diff + f3_diff

    #print("f_all_change: ",f_all_change)

    # If the fingers have only changed a small amount, we assume the object is grasped
    if f_all_change < 0.00005:
        #print("RETURN TRUE")
        return True
    else:
        #print("***************************f_all_change: ", f_all_change)
        #print("RETURN FALSE")
        return False

#def check_object_in_center(state):
# Define ranges of xa values to be interpolated
# Define center range based on interpolated x values
# Check if point is within center range
# -0.04 to -0.02, 0.02 to 0.04
#if -0.04 <= state[21] <= 0.04:
# Return True if object position is within center range
# Return False if object position is on outer edges

def GenerateExpertPID_JointVel(shape, episode_num, replay_buffer=None, save=True):
    env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    env.env.pid=True
    # episode_num = 10
    obs_label = []
    grasp_label = []
    action_label = []
    expert_success_x = np.array([])     # Successful object initial x-coordinates
    expert_success_y = np.array([])     # Successful object initial y-coordinates
    expert_fail_x = np.array([])     # Failed object initial x-coordinates
    expert_fail_y = np.array([])     # Files object initial y-coordinates
    all_timesteps = np.array([])     # Keeps track of all timesteps to determine average timesteps needed
    success_timesteps = np.array([]) # Successful episode timesteps count distribution
    fail_timesteps = np.array([])    # Failed episode timesteps count distribution
    hand_orientation_list = ["normal", "side", "top"]
    #shape_key = #["CubeS"]#, "CylinderB"] #, "Cube45S", "Vase2S", "Cone2S", "HourS", "BottleS", "TBottleS", "Cube45B", "Vase2B", "Cone2B", "HourB", "BottleB", "TBottleB"]
    #rotated = ["Yes"]
    expert_normal = ['expert', 'naive']
    print("----Generating {} expert episodes----".format(episode_num))
    for hand_orientation in hand_orientation_list:
        for expert in expert_normal:
            lift_success_list = np.zeros(episode_num-1)
            object_cord = np.zeros((episode_num-1, 3))
            orientation_list = np.zeros((episode_num-1, 3))
            finger_values_list = np.zeros((episode_num-1, 18))
            filepath = "./Data_with_noise/"+str(shape)+"/"+str(hand_orientation)+"/"+str(expert)+"/"
            for i in range(episode_num-1):
                print("****Shape: {} PID: {} Orientation: {} Episode {}: ".format(shape, expert, hand_orientation, i))
                prev_obs = None # State observation of the previous state
                ready_for_lift = False # Signals if ready for lift, from check_grasp()
                total_steps = 0
                obs, done = env.reset(shape_keys=shape, hand_orientation=hand_orientation, hand_rotation=True, counter=i), False
                # Sets number of timesteps per episode (counted from each step() call)
                env._max_episode_steps = 400
                obj_coords = env.get_obj_coords()

                controller = ExpertPIDController(obs)
                if replay_buffer != None:
                    replay_buffer.add_episode(1)
                while not done:

                    if prev_obs is None:  # None if on the first timestep
                        f_dist_old = None
                    else:
                        f_dist_old = prev_obs[9:17]
                    f_dist_new = obs[9:17]

                    obs_label.append(obs)
                    object_x_coord = obs[21] # Object x coordinate position
                    
                    #Expert Nudge controller strategy
                    if expert == 'expert':
                        action, grasp_label, ready_for_lift, f1_vels, f2_vels, f3_vels, wrist_vels = controller.NudgeController(prev_obs, obs, env.action_space, grasp_label)
                        # Do not lift until after 50 steps
                        if total_steps < 50:
                            action[0] = 0
                    elif expert == 'naive':
                        if naive_check_grasp(f_dist_old, f_dist_new) is True and total_steps > 50:
                            action = np.array([0.6, 0.15, 0.15, 0.15])
                        else:
                            action = np.array([0, 0.8, 0.8, 0.8])

                    scale = 1
                    #print("BEFORE action: ", action)
                    action = action * scale
                    #print("AFTER action: ", action,"\n")

                    action_label.append(action)
                    next_obs, reward, done, info = env.step(action)

                    if replay_buffer != None:
                        replay_buffer.add(obs[0:82], action, next_obs[0:82], reward, float(done))

                    # Once current timestep is over, update prev_obs to be current obs
                    if total_steps > 0:
                        prev_obs = obs
                    obs = next_obs
                    total_steps += 1

                    all_timesteps = np.append(all_timesteps,total_steps)

                    #print("Expert PID total timestep: ", total_steps)
                    lift_success=None
                    if (info["lift_reward"] > 0):
                        lift_success = 'success'
                        success_timesteps = np.append(success_timesteps, total_steps)
                    else:
                        lift_success = 'fail'
                        fail_timesteps = np.append(fail_timesteps, total_steps)
                    # if total_steps % 10 == 0:
                    #     env.render_img(episode_num=i, timestep_num=total_steps,obj_coords=str(obj_coords[0])+"_"+str(obj_coords[1]),final_episode_type=lift_success)

                    ret = add_heatmap_coords(expert_success_x, expert_success_y,expert_fail_x,expert_fail_y, obj_coords,info)
                    expert_success_x = ret[0]
                    expert_success_y = ret[1]
                    expert_fail_x = ret[2]
                    expert_fail_y = ret[3]
                    if replay_buffer != None:
                        replay_buffer.add_episode(0)

                #####################
                ##Code to Save Data##
                #####################
                if lift_success == 'success':
                    lift_success = 1
                else:
                    lift_success = 0
                lift_success_list[i] = lift_success
                object_cord[i] = obj_coords
                orientation_list[i] = env.wrist_orientation
                finger_values_list[i] = obs[:18]
            # Ensure file path is created if it doesn't exist
            coord_save_path = Path(filepath)
            coord_save_path.mkdir(parents=True, exist_ok=True)

            np.save(filepath+"lift_success_list", lift_success_list)
            np.save(filepath+"object_cord", object_cord)
            np.save(filepath+"orientation_list", orientation_list)
            np.save(filepath+"finger_values_list", finger_values_list)

    print("Final # of Successes: ", len(expert_success_x))
    print("Final # of Failures: ", len(expert_fail_x))

    print("Saving coordinates...")
    # Save coordinates
    # Folder to save heatmap coordinates
    expert_saving_dir = "./21_1_expert_plots"
    if not os.path.isdir(expert_saving_dir):
        os.mkdir(expert_saving_dir)

    np.save(expert_saving_dir + "/success_timesteps",success_timesteps)
    np.save(expert_saving_dir + "/fail_timesteps", fail_timesteps)
    np.save(expert_saving_dir + "/all_timesteps", all_timesteps)

    expert_total_x = np.append(expert_success_x, expert_fail_x)
    expert_total_y = np.append(expert_success_y, expert_fail_y)
    save_coordinates(expert_success_x, expert_success_y, expert_saving_dir + "/heatmap_train_success_new")
    save_coordinates(expert_fail_x, expert_fail_y, expert_saving_dir + "/heatmap_train_fail_new")
    save_coordinates(expert_total_x, expert_total_y, expert_saving_dir + "/heatmap_train_total_new")

    print("Plotting timestep distribution...")
    plot_timestep_distribution(success_timesteps,fail_timesteps,all_timesteps, expert_saving_dir)

    save = False
    print("Save is: ",str(save))
    save_filepath = None
    if save and replay_buffer is not None:
        print("Saving...")
        # Check and create directory
        expert_replay_saving_dir = "./expert_replay_data"
        if not os.path.isdir(expert_replay_saving_dir):
            os.mkdir(expert_replay_saving_dir)
        #data = {}
        #data["states"] = obs_label
        #data["grasp_success"] = grasp_label
        #data["action"] = action_label
        #data["total_steps"] = total_steps
        #file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
        ''' Different attempt to save data as current method gets overloaded
        filename = filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".sav"
        from sklearn.externals import joblib
        print("trying joblib...")
        joblib.dump(data, filename)
        '''
        #print("trying pickle...")
        #pickle.dump(data, file)
        #file.close()

        #data = {}
        #data["states"] = replay_buffer.state
        #data["action"] = replay_buffer.action
        #data["next_states"] = replay_buffer.next_state
        #data["reward"] = replay_buffer.reward
        #data["done"] = replay_buffer.not_done

        curr_save_dir = "Expert_data_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M")

        if not os.path.exists(os.path.join(expert_replay_saving_dir, curr_save_dir)):
            os.makedirs(os.path.join(expert_replay_saving_dir, curr_save_dir))

        save_filepath = expert_replay_saving_dir + "/" + curr_save_dir + "/"
        print("save_filepath: ",save_filepath)
        np.save(save_filepath + "state", replay_buffer.state)
        np.save(save_filepath + "action", replay_buffer.action)
        np.save(save_filepath + "next_state", replay_buffer.next_state)
        np.save(save_filepath + "reward", replay_buffer.reward)
        np.save(save_filepath + "not_done", replay_buffer.not_done)

        np.save(save_filepath + "episodes", replay_buffer.episodes) # Keep track of episode start/finish indexes
        np.save(save_filepath + "episodes_info", [replay_buffer.max_episode, replay_buffer.size, replay_buffer.episodes_count, replay_buffer.replay_ep_num])
        # max_episode: Maximum number of episodes, limit to when we remove old episodes
        # size: Full size of the replay buffer (number of entries over all episodes)
        # episodes_count: Number of episodes that have occurred (may be more than max replay buffer side)
        # replay_ep_num: Number of episodes currently in the replay buffer

        print("*** Saved replay buffer to location: ",save_filepath)
        print("In expert data: replay_buffer.size: ", replay_buffer.size)
    return replay_buffer, save_filepath
    
# def GenerateExpertPID_JointAngle():

def plot_timestep_distribution(success_timesteps=None, fail_timesteps=None, all_timesteps=None, expert_saving_dir=None):

    if all_timesteps is None:
        success_timesteps = np.load(expert_saving_dir + "/success_timesteps.npy")
        fail_timesteps = np.load(expert_saving_dir + "/fail_timesteps.npy")
        all_timesteps = np.load(expert_saving_dir + "/all_timesteps.npy")

    n_bins = 40
    # We can set the number of bins with the `bins` kwarg
    plt.hist(all_timesteps, bins=n_bins, color="g")
    plt.title("Total time steps distribution for all episodes (3x speed)", weight='bold')
    plt.xlabel('# of time steps per episode')
    plt.ylabel('# of episodes with the time step count')
    plt.xlim(0, 800)
    plt.savefig(expert_saving_dir + "/total_timestep_distribution")
    plt.clf()

    plt.hist(success_timesteps, bins=n_bins, color="b")
    plt.title("Time steps distribution for Successful episodes (3x speed)", weight='bold')
    plt.xlabel('# of time steps per episode')
    plt.ylabel('# of episodes with the time step count')
    plt.savefig(expert_saving_dir + "/success_timestep_distribution")
    plt.clf()

    plt.hist(fail_timesteps, bins=n_bins, color="r")
    plt.title("Time steps distribution for Failed episodes (3x speed)", weight='bold')
    plt.xlabel('# of time steps per episode')
    plt.ylabel('# of episodes with the time step count')
    plt.savefig(expert_saving_dir + "/fail_timestep_distribution")
    plt.clf()
'''
def plot_average_velocity(replay_buffer,num_timesteps):
    """ Plot the average velocity over a certain number of episodes """
    velocity_dir = "./expert_average_velocity"
    if not os.path.isdir(velocity_dir):
        os.mkdir(velocity_dir)

    #num_episodes = len(f1_vels)

    #plt.plot(np.arrange(len(f1_vels)), f1_vels)

    max_timesteps = 30
    timestep_vel_count = np.zeros(max_timesteps)
    wrist_avg_vels = np.zeros(max_timesteps)
    f1_avg_vels = np.zeros(max_timesteps)
    f2_avg_vels = np.zeros(max_timesteps)
    f3_avg_vels = np.zeros(max_timesteps)

    for episode_actions in replay_buffer.action:
        for timestep_idx in range(len(episode_actions)):
            timestep_vel_count[timestep_idx] += 1
            wrist_avg_vels[timestep_idx] = (wrist_avg_vels[timestep_idx] + episode_actions[timestep_idx][0]) / timestep_vel_count[timestep_idx]
            f1_avg_vels[timestep_idx] = (f1_avg_vels[timestep_idx] + episode_actions[timestep_idx][1]) / \
                                       timestep_vel_count[timestep_idx]
            f2_avg_vels[timestep_idx] = (f2_avg_vels[timestep_idx] + episode_actions[timestep_idx][2]) / \
                                       timestep_vel_count[timestep_idx]
            f3_avg_vels[timestep_idx] = (f3_avg_vels[timestep_idx] + episode_actions[timestep_idx][3]) / \
                                       timestep_vel_count[timestep_idx]

    num_episodes = len(replay_buffer.action)
    print("replay_buffer.action: ",replay_buffer.action)
    print("f1_avg_vels: ",f1_avg_vels)
    plt.plot(np.arange(num_timesteps), f1_avg_vels, color="r", label="Finger1")
    plt.plot(np.arange(num_timesteps), f2_avg_vels, color="b", label="Finger2")
    plt.plot(np.arange(num_timesteps), f3_avg_vels, color="g", label="Finger3")
    plt.plot(np.arange(num_timesteps), wrist_avg_vels, color="y", label="Wrist")
    plt.legend()

    plt.title("Average velocity over "+str(num_episodes)+" episodes", weight='bold')
    plt.xlabel('Timestep within an episode')
    plt.ylabel('Average Velocity at Timestep')
    #plt.savefig(velocity_dir + "/velocity_plot")
    #plt.clf()
    plt.show()
'''
def store_saved_data_into_replay(replay_buffer,filepath):

    print("#### Getting expert replay buffer from SAVED location: ",filepath)

    expert_state = np.load(filepath + "state.npy", allow_pickle=True).astype('object')
    expert_action = np.load(filepath + "action.npy", allow_pickle=True).astype('object')
    expert_next_state = np.load(filepath + "next_state.npy", allow_pickle=True).astype('object')
    expert_reward = np.load(filepath + "reward.npy", allow_pickle=True).astype('object')
    expert_not_done = np.load(filepath + "not_done.npy", allow_pickle=True).astype('object')

    expert_episodes = np.load(filepath + "episodes.npy", allow_pickle=True).astype('object')  # Keep track of episode start/finish indexes
    expert_episodes_info = np.load(filepath + "episodes_info.npy", allow_pickle=True)

    # Convert numpy array to list and set to replay buffer
    replay_buffer.state = expert_state.tolist()
    replay_buffer.action = expert_action.tolist()
    replay_buffer.next_state = expert_next_state.tolist()
    replay_buffer.reward = expert_reward.tolist()
    replay_buffer.not_done = expert_not_done.tolist()
    replay_buffer.episodes = expert_episodes.tolist()

    replay_buffer.max_episode = expert_episodes_info[0]
    replay_buffer.size = expert_episodes_info[1]
    replay_buffer.episodes_count = expert_episodes_info[2]
    replay_buffer.replay_ep_num = expert_episodes_info[3]

    # max_episode: Maximum number of episodes, limit to when we remove old episodes
    # size: Full size of the replay buffer (number of entries over all episodes)
    # episodes_count: Number of episodes that have occurred (may be more than max replay buffer side)
    # replay_ep_num: Number of episodes currently in the replay buffer

    #num_episodes = len(expert_state)
    num_episodes = replay_buffer.replay_ep_num
    print("num_episodes: ", num_episodes)

    return replay_buffer


def plot_timestep_distribution(success_timesteps=None, fail_timesteps=None, all_timesteps=None, expert_saving_dir=None):

    if all_timesteps is None:
        success_timesteps = np.load(expert_saving_dir + "/success_timesteps.npy")
        fail_timesteps = np.load(expert_saving_dir + "/fail_timesteps.npy")
        all_timesteps = np.load(expert_saving_dir + "/all_timesteps.npy")

    n_bins = 40
    # We can set the number of bins with the `bins` kwarg
    plt.hist(all_timesteps, bins=n_bins, color="g")
    plt.title("Total time steps distribution for all episodes (3x speed)", weight='bold')
    plt.xlabel('# of time steps per episode')
    plt.ylabel('# of episodes with the time step count')
    plt.xlim(0, 800)
    plt.savefig(expert_saving_dir + "/total_timestep_distribution")
    plt.clf()

    plt.hist(success_timesteps, bins=n_bins, color="b")
    plt.title("Time steps distribution for Successful episodes (3x speed)", weight='bold')
    plt.xlabel('# of time steps per episode')
    plt.ylabel('# of episodes with the time step count')
    plt.savefig(expert_saving_dir + "/success_timestep_distribution")
    plt.clf()

    plt.hist(fail_timesteps, bins=n_bins, color="r")
    plt.title("Time steps distribution for Failed episodes (3x speed)", weight='bold')
    plt.xlabel('# of time steps per episode')
    plt.ylabel('# of episodes with the time step count')
    plt.savefig(expert_saving_dir + "/fail_timestep_distribution")
    plt.clf()

# Command line
'''
# Collect entire sequence / trajectory
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so python train.py --num_episode 5000 --data_gen 1 --filename data_cube_5 

# Collect grasp data
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so python train.py --grasp_total_steps 10 --filename data_cube_5_10_07_19_1612 --grasp_filename data_cube_5_10_07_19_1612_grasp --grasp_validation 1 --data_gen 1

# Training
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so python train.py --grasp_validation 1 --filename data_cube_5_10_07_19_1612 --trained_model data_cube_5_trained_model --num_episode 5000
'''

# testing #
parser = argparse.ArgumentParser()
parser.add_argument("--shape", default="CubeS")
args = parser.parse_args()

replay_buffer, save_filepath = GenerateExpertPID_JointVel(shape = args.shape, episode_num = 4500)
#plot_timestep_distribution(success_timesteps=None, fail_timesteps=None, all_timesteps=None, expert_saving_dir="12_8_expert_test_3x_100ts")