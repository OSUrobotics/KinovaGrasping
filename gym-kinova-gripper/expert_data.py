#!/usr/bin/env python3

'''
Author : Yi Herng Ong
Date : 9/29/2019
Purpose : Supervised learning for near contact grasping strategy
'''

import os, sys
import numpy as np
import gym
from gym import wrappers  # Used for video rendering
import argparse
import pdb
import pickle
import datetime
# from NCS_nn import NCS_net, GraspValid_net
import torch
from copy import deepcopy
# from gen_new_env import gen_new_obj
import matplotlib.pyplot as plt
import utils
from pathlib import Path
import copy # For copying over coordinates
import math

# Import plotting code from other directory
plot_path = os.getcwd() + "/plotting_code"
sys.path.insert(1, plot_path)
from heatmap_coords import sort_and_save_heatmap_coords, coords_dict_to_array, save_coordinates
from trajectory_plot import plot_trajectory
from heatmap_plot import heatmap_actual_coords, get_hand_lines
from replay_stats_plot import actual_values_plot

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
        self.sampling_time = 0.15
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
        return self.action_range[1][-1] * (vel / 13.637)

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
        if self.x >= clipped_vel:  # if x accumulate to clipped vel, use clipped vel instead
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
            if self.vel[i + 1] < max_vel:
                # print("here")
                self.vel[i + 1] += 0.1 * self.vel[i + 1]
            else:
                self.vel[i + 1] = max_vel
        return np.array([wrist, self.vel[1], self.vel[2] * 0.7, self.vel[3] * 0.7])


def generate_lifting_data(env, total_steps, filename, grasp_filename):
    states = []
    label = []
    # file = open(filename + ".pkl", "rb")
    # data = pickle.load(file)
    # file.close()
    import timer
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
                obs, reward, _, _ = env.step(action)

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

    # lift
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
        dom_finger = env.env._get_obj_pose()[0]  # obj's position in x
        ini_dot_prod = env.env._get_dot_product(env.env._get_obj_pose())  #
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
            action_each_episode.append(jA_action)  # get joint angle as action
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
            if touch_dot_prod > 0.8:  # can only check dot product after fingers are making contact
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
        self.sampling_time = 15
        self.action_range = [action_space.low, action_space.high]

    def object_palm_normal_PID(self, dot_prod, velocities, velocity_scale=0.7):
        err = 1 - dot_prod
        diff = (err) / self.sampling_time
        vel = err * self.kp + diff * self.kd
        vel = (vel / 1.25)  # 1.25 means dot product equals to 1

        # Scale the velocity to the maximum velocity -
        # the PID was determined originally with a max of 0.8 rad/sec
        action = (vel / 0.8) * velocities["max_velocity"] * velocity_scale

        #action = max(action,0.3)

        controller_pid_metrics = {"action": action, "error": err, "diff": diff, "velocity": vel, "obj_dot_prod": dot_prod}

        return action, controller_pid_metrics

    def joint(self, dot_prod):
        err = 1 - dot_prod
        diff = (err) / self.sampling_time
        joint = err * self.kp + diff * self.kd
        action = (joint / 1.25) * 2  # 1.25 means dot product equals to 1
        return action

    def finger_object_distance_PID(self, finger_obj_distance, velocities, velocity_scale=9):
        """
        Determines the finger velocity based on the distance between the finger and the object.
        """
        target_distance = 0
        err = abs(target_distance - finger_obj_distance)
        diff = err / self.sampling_time
        vel = err * self.kp + diff * self.kd

        action = (vel/0.8) * velocities["max_velocity"] * velocity_scale

        # print("*** err: ", err)
        # print("*** diff: ", err)
        # print("*** vel: ", err)
        # print("*** action: ", action)
        controller_pid_metrics = {"action": action,"error": err, "diff": diff, "velocity": vel, "finger_obj_distance": finger_obj_distance}

        return action, controller_pid_metrics

    def get_angle_between_two_vectors(self, point1_start, point1_end, point2_start, point2_end):
        """Get angle between two vectors"""
        vector_1 = [point1_start[0]-point1_end[0], point1_start[1]-point1_end[1]]
        vector_2 = [point2_start[0]-point2_end[0], point2_start[1]-point2_end[1]]

        # Get the unit vectors
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)

        # Take the dot product between the two vectors
        dot_product = np.dot(unit_vector_1, unit_vector_2)

        # Determine the angle using the arc cosine
        angle = np.arccos(dot_product)

        return angle

    def finger_angle_based_pid(self,shape_name,timestep,obj_vectors,f1_vectors,f2_vectors,f3_vectors):
        """
        Get the velocity of all three fingers based on the current angle between the fingers and the object
        vs the desired finger and object angles wrt the palm center.
        """
        palm_center_coord = obj_vectors["palm_vec_points"]["palm_center"]
        point_along_palm_coord = obj_vectors["palm_vec_points"]["point_along_palm_center"]
        obj_coord = obj_vectors["geom_vec_points"]["geom_center_point"]
        finger1_coord = f1_vectors["geom_vec_points"]["geom_center_point"]
        finger2_coord = f2_vectors["geom_vec_points"]["geom_center_point"]
        finger3_coord = f3_vectors["geom_vec_points"]["geom_center_point"]

        # Object widths in meters
        object_widths = {"CubeS": 0.034, "CubeM": 0.041, "CubeB": 0.048, "CylinderM": 0.042,"Vase1M": 0.042}

        # AlphaOBJ angle between object and palm
        AlphaOBJ = self.get_angle_between_two_vectors(point1_start=palm_center_coord, point1_end=obj_coord, point2_start=palm_center_coord, point2_end=point_along_palm_coord)

        # AlphaL angle between obj and fingers (left)
        AlphaL = self.get_angle_between_two_vectors(point1_start=palm_center_coord, point1_end=finger1_coord, point2_start=palm_center_coord, point2_end=obj_coord)

        # AlphaR angle between obj and fingers (right)
        AlphaR = self.get_angle_between_two_vectors(point1_start=palm_center_coord, point1_end=finger2_coord, point2_start=palm_center_coord, point2_end=obj_coord)

        # AlphaL* desired angle at contact (AlphaL - atan2(width/2, dist to obj))
        dist_to_obj = 0.07 # meters
        half_width = object_widths[shape_name]/2
        target_Alpha = math.atan2(half_width, dist_to_obj)

        desired_AlphaL= target_Alpha

        # alphaR* desired angle at contact (AlphaR - atan2(width/2, dist to obj))
        desired_AlphaR = target_Alpha

        # DeltaMin Minimum angle change at contact
        DeltaMin = 0.017 * 2# 1 degree
        # DeltaMax Maximum angle change allowed
        # Max angle from finger to object is around 160 degrees / 30 timesteps
        DeltaMax = 0.017 * 5 #0.087 # 5 degrees

        # N - number of timesteps to desired contact
        N = 25 #30

        # DeltaDesiredL = DeltaMin + (alphaL - alphaL*) / N
        DeltaDesiredL = (AlphaL - desired_AlphaL)
        # DeltaDesiredR = DeltaMin + (alphaL - alphaL*) / N
        DeltaDesiredR = (AlphaR - desired_AlphaR)
        DeltaDesiredL = DeltaMin + DeltaDesiredL / N
        DeltaDesiredR = DeltaMin + DeltaDesiredR / N

        # Proportion of left angle change
        Prop = AlphaL / (AlphaL + AlphaR)

        if DeltaDesiredL > DeltaMax or DeltaDesiredR > DeltaMax:
            Decrease_perc = DeltaMax / max(DeltaDesiredL, DeltaDesiredR)
            DeltaDesiredL = DeltaDesiredL * Decrease_perc
            DeltaDesiredR = DeltaDesiredR * Decrease_perc

        #DeltaDesiredL = DeltaMin
        #DeltaDesiredR = DeltaMin

        """
        # if we have to cap one, then scale both by cap
        if DeltaDesiredL > DeltaMax and Prop >= 0.5:
            # Delta DesiredR has to be less than DeltaDesiredL
            DeltaDesiredL = DeltaMax
            DeltaDesiredR = min(DeltaMax, DeltaDesiredR * (1 - Prop))
        elif DeltaDesiredR > DeltaMax and Prop <= 0.5:
            # Delta DesiredL has to be less than DeltaDesiredR
            DeltaDesiredR = DeltaMax
            DeltaDesiredL = min(DeltaMax, DeltaDesiredL * (1 - Prop))
        """

        # Change in angle (rad) / timestep = change in angle (rad) / second
        # timestep (sec) = simulation timestep (sec) * # frames
        timestep_sec = 0.002 * 15 # 0.03 sec

        # radians / second
        f1_velocity = DeltaDesiredL / timestep_sec
        f2_velocity = DeltaDesiredR / timestep_sec
        f3_velocity = f2_velocity

        return f1_velocity, f2_velocity, f3_velocity

"""" NEW IMPLEMENTATION
AlphaOBJ angle between object and palm
AlphaL angle between obj and fingers (left)
AlphaR angle between obj and fingers (right)
AlphaL* desired angle at contact (AlphaL - atan2(width/2, dist to obj))
alphaR* desired angle at contact (AlphaR - atan2(width/2, dist to obj))

DeltaMin Minimum angle change at contact
DeltaMax Maximum angle change allowed
N - number of timesteps to desired contact

DeltaDesiredL = DeltaMin + (alphaL - alphaL*) / N
DeltaDesiredR = DeltaMin + (alphaR - alphaR*) / N

Prop = AlphaL/ (AlphaL + alphaR)
Alternative
If DeltaDesiredL > DeltaMax or DeltaDesireR > DeltaMax
     Decrease_perc = DeltaMax / max(DeltaDesiredL,DeltaDesireR) 
     DeltaDesiredL = DeltaDesiredL * Decrease_perc
     DeltaDesiredR = DeltaDesiredR * Decrease_perc

"""



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
        # self.init_obj_pose = self._sim.data.get_geom_xpos(object)
        self.init_dot_prod = states[81]  # dot product of object wrt palm
        self.f1_vels = []
        self.f2_vels = []
        self.f3_vels = []
        self.wrist_vels = []

    def _count(self):
        self.step += 1

    def center_action(self, constant_velocity, obj_dot_prod):
        """ Object is in a center location within the hand, so lift with constant velocity or adjust for lifting """
        print("CENTER action")

        # Check if change in object dot product to wrist center versus the initial dot product is greater than 0.01
        if abs(obj_dot_prod - self.init_dot_prod) > 0.01:
            # POST-Contact
            #print("CHECK 2: Obj dot product to wrist has changed more than 0.01")
            # Start lowering velocity of finger 2 and 3 so the balance of force is equal (no tipping)
            f1, f2, f3 = constant_velocity, (constant_velocity / 2), (constant_velocity / 2)
            print("POST-Contact")
        else:
            # PRE-Constact
            f1, f2, f3 = constant_velocity, constant_velocity, constant_velocity
            print("PRE-Contact")

        return np.array([f1, f2, f3])

    def right_action(self, pid, states, min_constant_velocity, obj_dot_prod, velocities):
        """ Object is in an extreme right-side location within the hand, so Finger 2 and 3 move the
        object closer to the center """
        print("RIGHT action")
        # Only Small change in object dot prod to wrist from initial position, must move more
        # Object has not moved much, we want the fingers to move closer to the object to move it
        if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
            """ PRE-contact """
            print("PRE-Contact")
            #print("CHECK 5: Only Small change in object dot prod to wrist, moving f2 & f3")
            f1 = min_constant_velocity
            f2 = pid.finger_object_distance_PID(obj_dot_prod, states[79], velocities)  # f2_dist dot product to object
            f3 = f2  # other double side finger moves at same speed
        else:
            """ POST-contact """
            # now finger-object distance has been changed a decent amount.
            #print("CHECK 6: Object dot prod to wrist has Changed More than 0.01")
            # Goal is 1 b/c obj_dot_prod is based on comparison of two normalized vectors
            if abs(1 - obj_dot_prod) > 0.01:
                print("POST-Contact, OUT of target range")
                #print("CHECK 7: Obj dot prod to wrist is > 0.01, so moving ALL f1, f2 & f3")
                # start to close the PID stuff
                f1 = min_constant_velocity  # frontal finger moves slightly
                f2 = pid.object_palm_normal_PID(obj_dot_prod, velocities)  # get PID velocity
                f3 = f2  # other double side finger moves at same speed
            else:  # goal is within 0.01 of being reached:
                #print("CHECK 8: Obj dot prod to wrist is Within reach of 0.01 or less, Move F1 Only")
                # start to close from the first finger
                print("POST-Contact, IN target range")
                f1 = pid.finger_object_distance_PID(obj_dot_prod, states[78], velocities)  # f1_dist dot product to object
                f2 = min_constant_velocity
                f3 = min_constant_velocity

        return np.array([f1, f2, f3])

    def left_action(self, pid, states, min_constant_velocity, obj_dot_prod, velocities):
        """ Object is in an extreme left-side location within the hand, so Finger 1 moves the
                object closer to the center """
        print("** LEFT action")
        # Only Small change in object dot prod to wrist from initial position, must move more
        if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
            """ PRE-contact """
            print("PRE-contact")
            #print("CHECK 11: Only Small change in object dot prod to wrist, moving F1")
            f1 = pid.finger_object_distance_PID(obj_dot_prod, states[78], velocities)  # f1_dist dot product to object
            f2 = min_constant_velocity
            f3 = min_constant_velocity
        else:
            """ POST-contact """
            # now finger-object distance has been changed a decent amount.
            #print("CHECK 12: Object dot prod to wrist has Changed More than 0.01")
            # Goal is 1 b/c obj_dot_prod is based on comparison of two normalized vectors
            if abs(1 - obj_dot_prod) > 0.01:
                print("POST-Contact, OUT of target range")
                #print("CHECK 13: Obj dot prod to wrist is > 0.01, so kep moving f1, f2 & f3")
                f1 = pid.object_palm_normal_PID(obj_dot_prod, velocities)
                f2 = min_constant_velocity
                f3 = min_constant_velocity
            else:
                print("POST-Contact, IN target range")
                # Goal is within 0.01 of being reached:
                #print("CHECK 14: Obj dot prod to wrist is Within reach of 0.01 or less, Move F2 & F3 Only")
                # start to close from the first finger
                # nudge with thumb
                f2 = pid.finger_object_distance_PID(obj_dot_prod, states[79], velocities)  # f2_dist dot product to object
                f3 = f2
                f1 = min_constant_velocity

        return np.array([f1, f2, f3])

    def PDController(self, states, action_space, velocities):
        """ Position-Dependent (PD) Controller that is dependent on the x-axis coordinate position of the object to 
        determine the individual finger velocities.
        """
        pid = PID(action_space) # Define pid controller
        obj_dot_prod = states[81] # Dot product of object wrt palm

        # Define action (finger velocities)
        f1 = 0.0  # far out finger, on single side
        f2 = 0.0  # double side finger - right top
        f3 = 0.0  # double side finger - right bottom
        wrist = 0.0

        # Velocity variables (for readability)
        constant_velocity = velocities["constant_velocity"]
        wrist_lift_velocity = velocities["wrist_lift_velocity"]
        finger_lift_velocity = velocities["finger_lift_velocity"]
        min_constant_velocity = velocities["min_constant_velocity"]
        min_velocity = velocities["min_velocity"]
        max_velocity = velocities["max_velocity"]

        # Note: only comparing initial X position of object. because we know
        # the hand starts at the same position every time (close to origin)

        # Check if the object is near the center area (less than x-axis 0.03)
        if abs(self.init_obj_pose) <= 0.03:
            #print("CHECK 1: Object is near the center")
            controller_action = self.center_action(constant_velocity, obj_dot_prod)
        else:
            #print("CHECK 3: Object is on extreme left OR right sides")
            # Object on right hand side, move 2-fingered side
            # Local representation: POS X --> object is on the RIGHT (two fingered) side of hand
            if self.init_obj_pose > 0.0:
                #print("CHECK 4: Object is on RIGHT side")
                controller_action = self.right_action(pid, states, min_constant_velocity, obj_dot_prod, velocities)

            # object on left hand side, move 1-fingered side
            # Local representation: NEG X --> object is on the LEFT (thumb) side of hand
            else:
                #print("CHECK 10: Object is on the LEFT side")
                controller_action = self.left_action(pid, states, min_constant_velocity, obj_dot_prod, velocities)

        #print("BEFORE Range check action: f1: {}, f2: {}, f3: {}".format(controller_action[0], controller_action[1], controller_action[2]))

        self._count()
        controller_action = check_vel_in_range(controller_action, velocities)

        #print("AFTER Final action: f1: {}, f2: {}, f3: {}\n".format(controller_action[0], controller_action[1], controller_action[2]))

        #print("f1: ", controller_action[0], " f2: ", controller_action[1], " f3: ", controller_action[2])
        self.f1_vels.append(f1)
        self.f2_vels.append(f2)
        self.f3_vels.append(f3)
        self.wrist_vels.append(wrist)

        return controller_action, self.f1_vels, self.f2_vels, self.f3_vels, self.wrist_vels

    def Stage_1_finger_obj_dist(self,pid,finger_obj_distance,velocities,velocity_scale=9):
        """
        Determines the finger velocity based on the finger-object distance based PID velocity
        """
        finger_velocity, finger_pid_metrics = pid.finger_object_distance_PID(finger_obj_distance, velocities, velocity_scale)

        return finger_velocity, finger_pid_metrics

    def Stage_1_finger_angle(self,pid,finger_name,shape_name,timestep,obj_vectors,f1_distance,f2_distance,f3_distance,f1_vectors,f2_vectors,f3_vectors):
        """
        Determines the finger velocity based on the angle between the finger and the object
        """
        f1_velocity, f2_velocity, f3_velocity = pid.finger_angle_based_pid(shape_name,timestep,obj_vectors,f1_vectors,f2_vectors,f3_vectors)

        f1_pid_metrics = {"action": f1_velocity, "finger_obj_distance": f1_distance}
        f2_pid_metrics = {"action": f2_velocity, "finger_obj_distance": f2_distance}
        f3_pid_metrics = {"action": f3_velocity, "finger_obj_distance": f3_distance}

        if finger_name == "finger_1":
            return f1_velocity, f1_pid_metrics
        elif finger_name == "finger_2":
            return f2_velocity, f2_pid_metrics
        elif finger_name == "finger_3":
            return f3_velocity, f3_pid_metrics

    def Stage_2(self,pid,obj_dot_prod,velocities,velocity_scale):
        """
        Determines the finger velocity based on the Object palm normal based PID
        """
        finger_velocity, finger_pid_metrics = pid.object_palm_normal_PID(obj_dot_prod, velocities,velocity_scale)

        return finger_velocity, finger_pid_metrics

    def Controller_A(self, env, states, timestep, action_space, velocities):
        """ Contact-Dependent (CD) Controller is a controller that is dependent on whether or not the object has come
        in contact with the fingers. If the object has not come in contact with the hand yet (PRE-Contact), then the
        TOUCH Veclocity PID determines each of the individual finger velocities. Otherwise, the object is considered
        POST-contact and the Velocity PID controller sets the individual finger velocities.
        """
        pid = PID(action_space) # Define pid controller
        contact_str = "Controller A\nStage 1"
        shape_name = env.random_shape

        # Get the dot product with respect to the palm center and a point along the y-axis
        obj_dot_prod, obj_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='object')
        f1_dot_prod, f1_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='f1_dist')
        f2_dot_prod, f2_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='f2_dist')
        f3_dot_prod, f3_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='f3_dist')

        f1_distance = math.sqrt(((f1_vectors["geom_vec_points"]["geom_center_point"][0] - obj_vectors["geom_vec_points"]["geom_center_point"][0]) ** 2) + ((f1_vectors["geom_vec_points"]["geom_center_point"][1] - obj_vectors["geom_vec_points"]["geom_center_point"][1]) ** 2))
        f2_distance = math.sqrt(((f2_vectors["geom_vec_points"]["geom_center_point"][0] - obj_vectors["geom_vec_points"]["geom_center_point"][0]) ** 2) + ((f2_vectors["geom_vec_points"]["geom_center_point"][1] - obj_vectors["geom_vec_points"]["geom_center_point"][1]) ** 2))
        f3_distance = math.sqrt(((f3_vectors["geom_vec_points"]["geom_center_point"][0] - obj_vectors["geom_vec_points"]["geom_center_point"][0]) ** 2) + ((f3_vectors["geom_vec_points"]["geom_center_point"][1] - obj_vectors["geom_vec_points"]["geom_center_point"][1]) ** 2))

        f1_velocity, f2_velocity, f3_velocity = pid.finger_angle_based_pid(shape_name,timestep,obj_vectors,f1_vectors,f2_vectors,f3_vectors)

        f1_pid_metrics = {"action": f1_velocity, "finger_obj_distance": f1_distance}
        f2_pid_metrics = {"action": f2_velocity, "finger_obj_distance": f2_distance}
        f3_pid_metrics = {"action": f3_velocity, "finger_obj_distance": f3_distance}

        # vectors = {"geom_vec_points": [palm_center, geom_center_point], "palm_vec_points": [palm_center, point_along_palm_center]}
        obj_hand_vectors = {"finger_1": {"vectors":f1_vectors,"finger-object_distance":f1_distance}, "finger_2": {"vectors":f2_vectors,"finger-object_distance":f2_distance}, "finger_3": {"vectors":f3_vectors,"finger-object_distance":f3_distance}, "object":{"vectors":obj_vectors,"finger-object_distance":None}}

        controller_action = np.array([f1_velocity,f2_velocity,f3_velocity])
        controller_pid_metrics = np.array([f1_pid_metrics,f2_pid_metrics,f3_pid_metrics])

        self._count()
        controller_action = check_vel_in_range(controller_action, velocities)

        #print("Final action: f1: {}, f2: {}, f3: {}\n".format(controller_action[0], controller_action[1], controller_action[2]))

        return controller_action, obj_hand_vectors, controller_pid_metrics, contact_str

    def Controller_B(self, env, states, timestep, action_space, velocities):
        """ Contact-Dependent (CD) Controller is a controller that is dependent on whether or not the object has come
        in contact with the fingers. If the object has not come in contact with the hand yet (PRE-Contact), then the
        TOUCH Veclocity PID determines each of the individual finger velocities. Otherwise, the object is considered
        POST-contact and the Velocity PID controller sets the individual finger velocities.
        """
        pid = PID(action_space) # Define pid controller
        if timestep == 1:
            self.init_dot_prod, _ = env.get_dot_product_wrt_to_palm_center(geom_name='object')

        # Get the dot product with respect to the palm center and a point along the y-axis
        obj_dot_prod, obj_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='object')
        f1_dot_prod, f1_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='f1_dist')
        f2_dot_prod, f2_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='f2_dist')
        f3_dot_prod, f3_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='f3_dist')
        obj_hand_vectors = {"finger_1": {"vectors":f1_vectors,"finger-object_distance":None}, "finger_2": {"vectors":f2_vectors,"finger-object_distance":None}, "finger_3": {"vectors":f3_vectors,"finger-object_distance":None}, "object":{"vectors":obj_vectors,"finger-object_distance":None}}

        f1_distance = math.sqrt(((f1_vectors["geom_vec_points"]["geom_center_point"][0] - obj_vectors["geom_vec_points"]["geom_center_point"][0]) ** 2) + ((f1_vectors["geom_vec_points"]["geom_center_point"][1] - obj_vectors["geom_vec_points"]["geom_center_point"][1]) ** 2))
        f2_distance = math.sqrt(((f2_vectors["geom_vec_points"]["geom_center_point"][0] - obj_vectors["geom_vec_points"]["geom_center_point"][0]) ** 2) + ((f2_vectors["geom_vec_points"]["geom_center_point"][1] - obj_vectors["geom_vec_points"]["geom_center_point"][1]) ** 2))
        f3_distance = math.sqrt(((f3_vectors["geom_vec_points"]["geom_center_point"][0] - obj_vectors["geom_vec_points"]["geom_center_point"][0]) ** 2) + ((f3_vectors["geom_vec_points"]["geom_center_point"][1] - obj_vectors["geom_vec_points"]["geom_center_point"][1]) ** 2))
        finger_obj_distances = {"finger_1": f1_distance, "finger_2": f2_distance, "finger_3": f3_distance}
        finger_obj_dist_sum = f1_distance + f2_distance + f3_distance

        contact_str = ""
        controller_action = np.array([])
        controller_pid_metrics = np.array([])
        shape_name = env.random_shape

        # Check if the object has moved
        object_has_moved = False
        if abs(obj_dot_prod - self.init_dot_prod) >= 0.01:
            object_has_moved = True

        # Determine PRE- or POST- contact stage for each finger
        for finger_name,finger_obj_distance in finger_obj_distances.items():
            # Object has NOT moved - PRE-CONTACT
            if object_has_moved is False:
                """ Stage 1: PRE-contact """
                #print("PRE-contact : ",finger_name)
                contact_str += finger_name + ": Stage 1\nPRE-Contact \n(object has NOT moved)\n"
                finger_velocity, finger_pid_metrics = self.Stage_1_finger_angle(pid,finger_name,shape_name,timestep,obj_vectors,f1_distance,f2_distance,f3_distance,f1_vectors,f2_vectors,f3_vectors)
                obj_hand_vectors[finger_name]["finger-object_distance"] = finger_obj_distance

            # Object has moved and finger is far away - PRE-CONTACT
            elif object_has_moved is True and finger_obj_distance > 0.05:
                """ Stage 1: PRE-contact """
                #print("PRE-contact: ",finger_name)
                contact_str += finger_name + ": Stage 1\nPRE-Contact \n(object has moved, finger too far)\n"
                finger_velocity, finger_pid_metrics = self.Stage_1_finger_angle(pid,finger_name,shape_name,timestep,obj_vectors,f1_distance,f2_distance,f3_distance,f1_vectors,f2_vectors,f3_vectors)
                obj_hand_vectors[finger_name]["finger-object_distance"] = finger_obj_distance

            # Object has moved and finger is close - POST-CONTACT
            elif object_has_moved is True and finger_obj_distance <= 0.05:
                #if finger_obj_distance >= 0.03:
                if finger_obj_distance <= 0.02:
                    velocity_scale = 0.3
                else:
                    velocity_scale = 0.7
                """ Stage 2: POST-contact, Nudge"""
                #print("POST-contact, Nudge: ", finger_name)
                contact_str += finger_name + ": Stage 2\nPOST-Contact - Nudge\n"
                finger_velocity, finger_pid_metrics = self.Stage_2(pid, obj_dot_prod, velocities, velocity_scale)
                finger_pid_metrics["finger_obj_distance"] = finger_obj_distance
                obj_hand_vectors[finger_name]["finger-object_distance"] = finger_obj_distance
                #else:
                # # Finger is within 0.03 m of object center - POST-CONTACT - FINAL STAGE
                # """ Stage 3: POST-contact, Final stage, Target zone"""
                # #print("POST-contact, Final stage, Target zone: ", finger_name)
                # contact_str += finger_name + ": Stage 1\nPOST-Contact - Final\n"
                # # Increase the velocity scale when the finger is close so the finger doesn't go too slow
                # finger_velocity, finger_pid_metrics = self.Stage_1_finger_angle(pid, finger_name, shape_name, timestep, obj_vectors, f1_distance, f2_distance, f3_distance, f1_vectors, f2_vectors, f3_vectors)
                # obj_hand_vectors[finger_name]["finger-object_distance"] = finger_obj_distance

                #finger_velocity = max(finger_velocity, 0.5)
                finger_pid_metrics["action"] = finger_velocity

            controller_action = np.append(controller_action,finger_velocity)
            controller_pid_metrics = np.append(controller_pid_metrics,finger_pid_metrics)

        self._count()
        controller_action = check_vel_in_range(controller_action, velocities)

        #print("Final action: f1: {}, f2: {}, f3: {}\n".format(controller_action[0], controller_action[1], controller_action[2]))

        return controller_action, obj_hand_vectors, controller_pid_metrics, contact_str


def plot_controller_metrics(controller_output,velocities,all_saving_dirs):
    # Plot the output of each metric
    # Dot product should decrease over time, same with error
    for episode_num in controller_output.keys():
        # Plot all the values over the course of the episode
        # distance/velocity per finger
        # obj_hand_vectors = {"finger_1": {"vectors":f1_vectors,"finger-object_distance":f1_distance}, "finger_2": {"vectors":f2_vectors,"finger-object_distance":f2_distance}, "finger_3": {"vectors":f3_vectors,"finger-object_distance":f3_distance}}
        ep_saving_dir = all_saving_dirs["output_dir"]+"/ep_"+str(episode_num)
        #new_path = Path(ep_saving_dir)
        #new_path.mkdir(parents=True, exist_ok=True)

        for finger, finger_idx in zip(controller_output[episode_num]["obj_hand_vectors"][0].keys(), [0,1,2]):
            # Plot the actions over time
            controller_actions = [action_value[finger_idx] for action_value in controller_output[episode_num]["actions"]]
            finger_obj_dists = [dist[finger]["finger-object_distance"] for dist in controller_output[episode_num]["obj_hand_vectors"]]
            #pid_error = [controller_metric[finger_idx]["error"] for controller_metric in controller_output[episode_num]["controller_pid_metrics"]]
            #obj_dot_prod = [controller_metric[finger_idx]["obj_dotprod"] for controller_metric in controller_output[episode_num]["controller_pid_metrics"]]
            #inger_dot_prod = [controller_metric[finger_idx]["finger_dotprod"] for controller_metric in controller_output[episode_num]["controller_pid_metrics"]]

            # Plot the velocity over time
            axes_limits = {"x_min": 0, "x_max": len(controller_actions), "y_min": 0, "y_max": velocities["max_velocity"]}
            actual_values_plot(metrics_arr=[controller_actions], episode_idx=0, label_name="Finger " + str(finger_idx + 1) + " Velocity",
                               y_metric_name="Action Output: Finger " + str(finger_idx + 1) + " Velocity",
                               metric_name="Timestep",axes_limits=axes_limits, saving_dir=ep_saving_dir)

            # Plot the object dot prod over time
            #axes_limits = {"x_min": 0, "x_max": len(obj_dot_prod), "y_min": min(obj_dot_prod),
            #               "y_max": max(obj_dot_prod)}
            #actual_values_plot(metrics_arr=[obj_dot_prod], episode_idx=0, label_name="object-palm dot prod",
            #                   metric_name="Timestep", y_metric_name="Object-palm dot product", axes_limits=axes_limits,
            #                   saving_dir=ep_saving_dir)

            # Plot the finger dot prod over time
            #axes_limits = {"x_min": 0, "x_max": len(finger_dot_prod), "y_min": min(finger_dot_prod),
            #               "y_max": max(finger_dot_prod)}
            #actual_values_plot(metrics_arr=[finger_dot_prod], episode_idx=0, label_name=str(finger) + "-palm dot prod",
            #                   metric_name="Timestep", y_metric_name=str(finger) + "-palm dot product", axes_limits=axes_limits,
            #                   saving_dir=ep_saving_dir)

            # Plot the error over time
            """
            axes_limits = {"x_min": 0, "x_max": len(pid_error), "y_min": min(pid_error),
                           "y_max": max(pid_error)}
            actual_values_plot(metrics_arr=[pid_error], episode_idx=0, label_name=str(finger) + "-object PID error",
                               metric_name="Timestep", y_metric_name=str(finger) + "-object PID error", axes_limits=axes_limits,
                               saving_dir=ep_saving_dir)
            """
            # Plot the finger-object distance over time
            axes_limits = {"x_min": 0, "x_max": len(finger_obj_dists), "y_min": min(finger_obj_dists),
                           "y_max": max(finger_obj_dists)}
            actual_values_plot(metrics_arr=[finger_obj_dists], episode_idx=0, label_name=str(finger) + "-object distance",
                               metric_name=str(finger) + "-object distance", y_metric_name="Timestep", axes_limits=axes_limits,
                               saving_dir=ep_saving_dir)

            # Plot the finger-object distance vs velocity over time
            axes_limits = {"x_min": min(finger_obj_dists), "x_max": max(finger_obj_dists), "y_min": 0, "y_max": velocities["max_velocity"]}
            actual_values_plot(metrics_arr=[finger_obj_dists], y_axis_metrics=controller_actions, episode_idx=0, label_name=str(finger) + "-object distance",
                               metric_name=str(finger) + "-object distance", y_metric_name="Velocity", axes_limits=axes_limits,
                               saving_dir=ep_saving_dir)


def check_vel_in_range(action, velocities):
    """ Checks that each of the finger/wrist velocies values are in range of min/max values """
    for i in range(len(action)):
        if action[i] < velocities["min_velocity"]:
            action[i] = velocities["min_velocity"]
        elif action[i] > velocities["max_velocity"]:
            action[i] = velocities["max_velocity"]

    return action


def GenerateTestPID_JointVel(obs, env):
    controller = ExpertPIDController(obs)
    action = controller.PDController(obs, env.action_space)
    return action


def check_pid_grasp(f_dist_old, f_dist_new):
    """
    Uses the current change in x,y position of the distal finger tips, summed over all fingers to determine if
    the object is grasped (fingers must have only changed in position over a tiny amount to be considered done).
    @param: f_dist_old: Distal finger tip x,y,z coordinate values from previous timestep
    @param: f_dist_new: Distal finger tip x,y,z coordinate values from current timestep
    """

    # Initial check to see if previous state has been set
    if f_dist_old is None:
        return 0
    sampling_time = 15

    # Change in finger 1 distal x-coordinate position
    f1_change = abs(f_dist_old[0] - f_dist_new[0])
    f1_diff = f1_change / sampling_time

    # Change in finger 2 distal x-coordinate position
    f2_change = abs(f_dist_old[3] - f_dist_new[3])
    f2_diff = f2_change / sampling_time

    # Change in finger 3 distal x-coordinate position
    f3_change = abs(f_dist_old[6] - f_dist_new[6])
    f3_diff = f3_change / sampling_time

    # Sum of changes in distal fingers
    f_all_change = f1_diff + f2_diff + f3_diff

    #print("f_all_change: ", f_all_change)

    # If the fingers have only changed a small amount, we assume the object is grasped
    if f_all_change < 0.0002:
        return [1, f_all_change]
    else:
        return [0, f_all_change]


def NaiveController(velocities,env=None):
    """ Move fingers at a constant speed, return action """
    # Get the dot product with respect to the palm center and a point along the y-axis
    obj_dot_prod, obj_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='object')
    f1_dot_prod, f1_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='f1_dist')
    f2_dot_prod, f2_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='f2_dist')
    f3_dot_prod, f3_vectors = env.get_dot_product_wrt_to_palm_center(geom_name='f3_dist')

    f1_distance = math.sqrt(((f1_vectors["geom_vec_points"]["geom_center_point"][0] - obj_vectors["geom_vec_points"]["geom_center_point"][0]) ** 2) + ((f1_vectors["geom_vec_points"]["geom_center_point"][1] - obj_vectors["geom_vec_points"]["geom_center_point"][1]) ** 2))
    f2_distance = math.sqrt(((f2_vectors["geom_vec_points"]["geom_center_point"][0] - obj_vectors["geom_vec_points"]["geom_center_point"][0]) ** 2) + ((f2_vectors["geom_vec_points"]["geom_center_point"][1] - obj_vectors["geom_vec_points"]["geom_center_point"][1]) ** 2))
    f3_distance = math.sqrt(((f3_vectors["geom_vec_points"]["geom_center_point"][0] - obj_vectors["geom_vec_points"]["geom_center_point"][0]) ** 2) + ((f3_vectors["geom_vec_points"]["geom_center_point"][1] - obj_vectors["geom_vec_points"]["geom_center_point"][1]) ** 2))

    # By default, close all fingers at a constant speed
    action = np.array([velocities["constant_velocity"], velocities["constant_velocity"], velocities["constant_velocity"]])

    f1_pid_metrics = {"action": action[0], "finger_obj_distance": f1_distance}
    f2_pid_metrics = {"action": action[1], "finger_obj_distance": f2_distance}
    f3_pid_metrics = {"action": action[2], "finger_obj_distance": f3_distance}
    controller_pid_metrics = np.array([f1_pid_metrics, f2_pid_metrics, f3_pid_metrics])

    obj_hand_vectors = {"finger_1": {"vectors": f1_vectors, "finger-object_distance": f1_distance},
                        "finger_2": {"vectors": f2_vectors, "finger-object_distance": f2_distance},
                        "finger_3": {"vectors": f3_vectors, "finger-object_distance": f3_distance},
                        "object": {"vectors": obj_vectors, "finger-object_distance": None}}

    return action, obj_hand_vectors, controller_pid_metrics

def BellShapedController(velocities, timestep):
    """ Move fingers at a constant speed, return action """

    bell_curve_velocities = [0.202, 0.27864, 0.35046, 0.41696, 0.47814, 0.534, 0.58454, 0.62976, 0.66966, 0.70424, 0.7335, 0.75744, 0.77606, 0.78936, 0.79734, 0.8, 0.79734, 0.78936, 0.77606, 0.75744, 0.7335, 0.70424, 0.66966, 0.62976, 0.58454, 0.534, 0.47814, 0.41696, 0.35046, 0.27864, 0.2015]

    # Determine the finger velocities by increasing and decreasing the values with a constant acceleration
    finger_velocity = bell_curve_velocities[timestep]

    # By default, close all fingers at a constant speed
    action = np.array([finger_velocity, finger_velocity, finger_velocity])

    return action


def get_action(obs, controller, env, episode_num, timestep, velocities, pid_mode, all_saving_dirs):
    """ Get action based on controller (Naive, position-dependent, combined interpolation)
        obs: Current state observation
        controller: Initialized expert PID controller
        env: Current Mujoco environment needed for expert PID controller
        return action: np.array([wrist, f1, f2, f3]) (velocities in rad/sec)
    """
    object_x_coord = obs[21]  # Object x coordinate position

    # By default, action is set to close fingers at a constant velocity
    controller_action = np.array([velocities["constant_velocity"], velocities["constant_velocity"], velocities["constant_velocity"]])

    # NAIVE CONTROLLER: Close all fingers at a constant speed
    if pid_mode == "naive":
        controller_action, obj_hand_vectors, controller_pid_metrics = NaiveController(velocities,env)

    # POSITION-DEPENDENT CONTROLLER: Only move fingers based on object x-coord position within hand
    elif pid_mode == "position-dependent":
        controller_action, f1_vels, f2_vels, f3_vels, wrist_vels = controller.PDController(obs, env.action_space, velocities)

    elif pid_mode == "bell-shaped":
        controller_action = BellShapedController(velocities, timestep)

    elif pid_mode == "controller_a":
        controller_action, obj_hand_vectors, controller_pid_metrics, contact_str = controller.Controller_A(env, obs, timestep, env.action_space, velocities)
        finger_coords = obs[0:17]
        wrist_coords = obs[18:20]
        hand_lines = get_hand_lines("local", wrist_coords, finger_coords)
        title = "ep_"+str(episode_num)+"ts_"+str(timestep)

        finger_ob_dist_sum = obj_hand_vectors["finger_1"]["finger-object_distance"] + obj_hand_vectors["finger_2"]["finger-object_distance"] + obj_hand_vectors["finger_3"]["finger-object_distance"]
        finger_obj_dist_str = "Finger-Object distances:\nfinger 1: {:.3f} m\nfinger 2: {:.3f} m\nfinger 3: {:.3f} m\nSum: {:.3f} m".format(obj_hand_vectors["finger_1"]["finger-object_distance"],obj_hand_vectors["finger_2"]["finger-object_distance"],obj_hand_vectors["finger_3"]["finger-object_distance"],finger_ob_dist_sum)
        obj_coords = env.get_obj_coords()
        contact_str += "\nObject coords: \n({:.3f},{:.3f})\n".format(obj_coords[0],obj_coords[1])
        velocity_str = "Finger Velocities:\nFinger 1 velocity: {:.3f}\nFinger 2 velocity: {:.3f}\nFinger 3 velocity: {:.3f}\n".format(controller_action[0],controller_action[1],controller_action[2])
        plot_str = [contact_str,velocity_str,finger_obj_dist_str]
        #if episode_num <= 10:
        #    heatmap_actual_coords(total_x=obs[21], total_y=obs[22], vector_lines=obj_hand_vectors,hand_lines=hand_lines, state_rep="local", plot_title=title, fig_filename=title+".png", plot_str=plot_str,saving_dir=all_saving_dirs["output_dir"]+"/ep_"+str(episode_num)+"/")

    elif pid_mode == "controller_b":
        controller_action, obj_hand_vectors, controller_pid_metrics, contact_str = controller.Controller_B(env, obs, timestep, env.action_space, velocities)
        finger_coords = obs[0:17]
        wrist_coords = obs[18:20]
        hand_lines = get_hand_lines("local", wrist_coords, finger_coords)
        title = "ep_"+str(episode_num)+"ts_"+str(timestep)

        finger_ob_dist_sum = obj_hand_vectors["finger_1"]["finger-object_distance"] + obj_hand_vectors["finger_2"]["finger-object_distance"] + obj_hand_vectors["finger_3"]["finger-object_distance"]
        finger_obj_dist_str = "Finger-Object distances:\nfinger 1: {:.3f} m\nfinger 2: {:.3f} m\nfinger 3: {:.3f} m\nSum: {:.3f} m".format(obj_hand_vectors["finger_1"]["finger-object_distance"],obj_hand_vectors["finger_2"]["finger-object_distance"],obj_hand_vectors["finger_3"]["finger-object_distance"],finger_ob_dist_sum)
        obj_coords = env.get_obj_coords()
        contact_str += "\nObject coords: \n({:.3f},{:.3f})\n".format(obj_coords[0],obj_coords[1])
        velocity_str = "Finger Velocities:\nFinger 1 velocity: {:.3f}\nFinger 2 velocity: {:.3f}\nFinger 3 velocity: {:.3f}\n".format(controller_action[0],controller_action[1],controller_action[2])
        plot_str = [contact_str,velocity_str,finger_obj_dist_str]
        if episode_num <= 10:
            heatmap_actual_coords(total_x=obs[21], total_y=obs[22], vector_lines=obj_hand_vectors,hand_lines=hand_lines, state_rep="local", plot_title=title, fig_filename=title+".png", plot_str=plot_str,saving_dir=all_saving_dirs["output_dir"]+"/ep_"+str(episode_num)+"/")

    # COMBINED CONTROLLER: Interpolate Naive and Position-Dependent controller output based on object x-coord position within hand
    else:
        # If object x position is on outer edges, do expert pid
        if object_x_coord < -0.04 or object_x_coord > 0.04:
            # Expert Nudge controller strategy
            controller_action, f1_vels, f2_vels, f3_vels, wrist_vels = controller.PDController(obs, env.action_space, velocities)

        # Object x position within the side-middle ranges, interpolate expert/naive velocity output
        elif -0.04 <= object_x_coord <= -0.02 or 0.02 <= object_x_coord <= 0.04:
            # Interpolate between naive and expert velocities
            # position-dependent controller action (finger velocity based on object location within hand)
            expert_action, f1_vels, f2_vels, f3_vels, wrist_vels = controller.PDController(obs, env.action_space, velocities)
            # Naive controller action (fingers move at constant velocity)
            naive_action = NaiveController(velocities)

            # Interpolate finger velocity values between position-dependent and Naive action output
            finger_vels = np.interp(np.arange(0, 3), naive_action, expert_action)

            controller_action = np.array([finger_vels[0], finger_vels[1], finger_vels[2]])

        # Object x position is within center area, so use naive controller
        else:
            # Naive controller action (fingers move at constant velocity)
            controller_action = NaiveController(velocities)

    #print("**** action: ",action)

    return controller_action, obj_hand_vectors, controller_pid_metrics


def set_action_str(action, num_good_grasps, obj_local_pos, obs, reward, naive_ret, info):
    """ Set string to show over simulation rendering for further context into finger/object movements, rewards,
        and whether the controller has signaled it is ready to lift.
    """
    velocity_str = "Wrist: " + str(action[0]) + "\nFinger1: " + str(action[1]) + "\nFinger2: " + str(
        action[2]) + "\nFinger3: " + str(action[3])
    lift_status_str = "\nnum_good_grasps: " + str(num_good_grasps) + "\nf_all_change: " + str(naive_ret)
    obj_str = "\nObject Local x,y: (" + str(obj_local_pos[0]) + ", " + str(obj_local_pos[1]) + ") " + "\nobject center height: " + str(obs[23])
    reward_str = "\ntimestep reward: " + str(reward) + "\nfinger reward: " + str(info["finger_reward"]) + \
                 "\ngrasp reward: " + str(info["grasp_reward"]) + "\nlift reward: " + str(info["lift_reward"])

    action_str = velocity_str + lift_status_str + obj_str + reward_str

    return action_str


def GenerateExpertPID_JointVel(episode_num, requested_shapes, requested_orientation, all_saving_dirs, replay_buffer=None, with_grasp=False, with_noise=False, save=True, render_imgs=False, pid_mode="combined"):
    """ Generate expert data based on Expert PID and Naive PID controller action output.
    episode_num: Number of episodes to generate expert data for
    replay_buffer: Replay buffer to be passed in (set to None for testing purposes)
    save: set to True if replay buffer data is to be saved
    """
    env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    env.env.pid = True

    # Use to test plotting the episode trajectory
    plot_ep_trajectory = None

    success_coords = {"x": [], "y": [], "orientation": []}
    fail_coords = {"x": [], "y": [], "orientation": []}
    # hand orientation types: NORMAL, Rotated (45 deg), Top (90 deg)

    all_timesteps = np.array([])  # Keeps track of all timesteps to determine average timesteps needed
    success_timesteps = np.array([])  # Successful episode timesteps count distribution
    fail_timesteps = np.array([])  # Failed episode timesteps count distribution
    datestr = datetime.datetime.now().strftime("%m_%d_%y_%H%M") # used for file name saving

    # Wrist lifting velocity
    wrist_lift_action = 1 # radians/sec

    print("----Generating {} expert episodes----".format(episode_num))
    print("Using PID MODE: ", pid_mode)

    # Beginning of episode loop
    for i in range(episode_num):
        print("PID ", i)
        # Fill training object list using latin square
        if env.check_obj_file_empty("objects.csv") or episode_num is 0:
            env.Generate_Latin_Square(episode_num, "objects.csv", shape_keys=requested_shapes)
        obs, done = env.reset(shape_keys=requested_shapes,hand_orientation=requested_orientation, with_noise=with_noise), False

        # Record initial coordinate file path once shapes are generated
        coord_filepath = env.get_coords_filename()
        prev_obs = None         # State observation of the previous state
        num_good_grasps = 0      # Counts the number of good grasps (Number of times check_grasp() has returned True)
        num_consistent_grasps = 1   # Number of RL steps needed
        total_steps = 0             # Total RL timesteps passed within episode
        action_str = "Initial time step" # Set the initial render output string
        env._max_episode_steps = 30 # Sets number of timesteps per episode (counted from each step() call)
        obj_coords = env.get_obj_coords()
        # Local coordinate conversion
        obj_local = np.append(obj_coords,1)
        obj_local = np.matmul(env.Tfw,obj_local)
        obj_local_pos = obj_local[0:3]

        controller = ExpertPIDController(obs)   # Initiate expert PID controller

        # Record episode starting index if replay buffer is passed in
        if replay_buffer is not None:
            replay_buffer.add_episode(1)
            # Add orientation noise to be recorded by replay buffer
            orientation_idx = env.get_orientation_idx()
            replay_buffer.add_orientation_idx_to_replay(orientation_idx)

        # Beginning of RL time steps within the current episode
        while not done:
            # Distal finger x,y,z positions f1_dist, f2_dist, f3_dist
            if prev_obs is None:  # None if on the first RL episode time step
                f_dist_old = None
            else:
                f_dist_old = prev_obs[9:17] # Track the changes in distal finger tip positions
            f_dist_new = obs[9:17]

            ### READY FOR LIFTING CHECK ###
            min_lift_timesteps = 10  # Number of time steps that must occur before attempting to lift object
            lift_check = False  # Check whether we have had enough good grasps and meet lifting requirements

            # Check if ready to lift based on distal finger tip change in movement
            grasp_check = check_pid_grasp(f_dist_old, f_dist_new)
            # (1) True if within distal finger movement is <= threshold (ready for lifting)
            if grasp_check is True:
                num_good_grasps += 1  # Increment the number of consecutive good grasps

            # Check if we have been in a good grasp position for num_consistent_grasps time steps
            if total_steps > min_lift_timesteps and num_good_grasps >= num_consistent_grasps:
                lift_check = True
            ### END OF READY FOR LIFTING CHECK ###

            # Get action based on the selected controller (Naive, position-dependent, combined Interpolation
            controller_action = get_action(obs, lift_check, controller, env, pid_mode)

            # From the controllers we get the iniividual finger velocities, then lift with the same wrist velocity
            if lift_check is True:
                wrist_action = np.array([wrist_lift_action])
                action = np.concatenate((wrist_action, controller_action))
            else:
                # No movement in the wrist if we are not ready for lifting (0 rad/sec)
                wrist_action = np.array([0])
                action = np.concatenate((wrist_action, controller_action))

            # Take action (Reinforcement Learning step)
            env.set_with_grasp_reward(with_grasp)
            next_obs, reward, done, info = env.step(action)

            # Set the info to be displayed in episode rendering based on current hand/object status
            action_str = set_action_str(action, num_good_grasps, obj_local_pos, obs, reward, naive_ret, info)

            # Render image from current episode
            if render_imgs is True:
                if total_steps % 1 == 0:
                    env.render_img(saving_dir=all_saving_dirs["output_dir"], text_overlay=action_str, episode_num=i,
                                   timestep_num=total_steps,
                                   obj_coords=obj_local_pos)
                else:
                    env._viewer = None

            # Add experience to replay buffer
            if replay_buffer is not None and not lift_check:
                # Only recording the controller's actions (finger velocities) within the replay buffer
                replay_buffer.add(obs[0:82], controller_action, next_obs[0:82], reward, float(done))

            if lift_check and done:
                replay_buffer.replace(reward, done)
                # print ("#######REWARD#######", reward)

            # Once current timestep is over, update prev_obs to be current obs
            if total_steps > 0:
                prev_obs = obs
            obs = next_obs
            total_steps += 1

        all_timesteps = np.append(all_timesteps, total_steps)

        success = 0
        if (info["lift_reward"] > 0):
            lift_success = 'success'
            success = 1
            success_timesteps = np.append(success_timesteps, total_steps)
        else:
            lift_success = 'fail'
            fail_timesteps = np.append(fail_timesteps, total_steps)

        # print("!!!!!!!!!!###########LIFT REWARD:#######!!!!!!!!!!!", info["lift_reward"])
        if render_imgs is True:
            # Set the info to be displayed in episode rendering based on current hand/object status
            action_str = set_action_str(action, num_good_grasps, obj_local_pos, obs, reward, naive_ret, info)

            if total_steps % 1 == 0:
                image = env.render_img(saving_dir=all_saving_dirs["output_dir"],text_overlay=action_str, episode_num=i, timestep_num=total_steps,
                               obj_coords=obj_local_pos, final_episode_type=success)

        # Add heatmap coordinates
        orientation = env.get_orientation()
        ret = add_heatmap_coords(success_coords, fail_coords, orientation, obj_local_pos, success)
        success_coords = copy.deepcopy(ret["success_coords"])
        fail_coords = copy.deepcopy(ret["fail_coords"])

        if replay_buffer is not None:
            replay_buffer.add_episode(0)
            episode_length = replay_buffer.episodes[i][1] # Ending index of the final episode time step

            # If the episode contains time step experience, plot the trajectory
            if episode_length > 0 and plot_ep_trajectory is not None:
                plot_trajectory(replay_buffer.state[i], replay_buffer.action[i], i, plot_ep_trajectory)

    num_success = len(success_coords["x"])
    num_fail = len(fail_coords["x"])
    print("Final # of Successes: ", num_success)
    print("Final # of Failures: ", num_fail)
    print("Shapes: ", requested_shapes)

    shapes_str = ""
    if isinstance(requested_shapes, list) is True:
        for shape in requested_shapes:
            shapes_str += str(shape) + "_"
        shapes_str = shapes_str[:-1]
    else:
        shapes_str = str(requested_shapes)

    grasp_str = "no_grasp"
    if with_grasp is True:
        grasp_str = "with_grasp"

    print("Saving coordinates...")
    # Save coordinates
    # Directory for x,y coordinate heatmap data
    # STEPH TEST NO NOISE
    expert_saving_dir = all_saving_dirs["saving_dir"] #"expert_replay_data/"+grasp_str+"/"+str(pid_mode)+"/"+str(shapes_str) +"/"+ str(requested_orientation)
    expert_output_saving_dir = all_saving_dirs["output_dir"] #expert_saving_dir + "/output"
    expert_replay_saving_dir = all_saving_dirs["replay_buffer"] #expert_saving_dir + "/replay_buffer"
    heatmap_saving_dir = all_saving_dirs["heatmap_dir"] #expert_output_saving_dir + "/heatmap/expert"

    print("\n--------- Saving Directories --------------")
    print("Main saving directory: ", expert_saving_dir)
    print("Output (data, plots) saved to: ", expert_output_saving_dir)
    print("Heatmap data saved to: ", heatmap_saving_dir)

    # Filter heatmap coords by success/fail, orientation type, and save to appropriate place
    filter_heatmap_coords(success_coords, fail_coords, None, heatmap_saving_dir)

    if save and replay_buffer is not None:
        print("\nSaving replay buffer...")

        # data = {}
        # file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
        ''' Different attempt to save data as current method gets overloaded
        filename = filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".sav"
        from sklearn.externals import joblib
        print("trying joblib...")
        joblib.dump(data, filename)
        '''
        # print("trying pickle...")
        # pickle.dump(data, file)
        # file.close()

        # data = {}
        # data["states"] = replay_buffer.state
        # data["action"] = replay_buffer.action
        # data["next_states"] = replay_buffer.next_state
        # data["reward"] = replay_buffer.reward
        # data["done"] = replay_buffer.not_done

        save_filepath = replay_buffer.save_replay_buffer(expert_replay_saving_dir)

        # Output info file text
        name_text = "PID MODE: " + str(pid_mode) + ", Num Grasp trials: " + str(episode_num) + "\nDate: {}".format(datetime.datetime.now().strftime("%m_%d_%y_%H%M"))
        success_text = "\n\nFinal # of Successes: " + str(num_success) + "\nFinal # of Failures: " + str(num_fail) + "\n"
        shapes_text = "Shapes: " + str(requested_shapes) + "\n\n"
        output_dir_text = "Saved replay buffer to location: "+ save_filepath + "\nreplay_buffer.replay_ep_num (# episodes): " + str(replay_buffer.replay_ep_num) + "\nreplay_buffer.size (# trajectories): " + str(replay_buffer.size) + "\nOutput data saved at: " + str(expert_saving_dir)

        text = name_text + success_text + shapes_text + output_dir_text

        print("Expert Replay Buffer Saved to: ", save_filepath)
        print("---- Replay Buffer Info -----")
        print("# Episodes: ", replay_buffer.replay_ep_num)
        print("# Trajectories: ", replay_buffer.size)

    return replay_buffer, save_filepath, expert_saving_dir, text, num_success, (num_success+num_fail), coord_filepath


# Command line
'''
# Collect entire sequence / trajectory
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so python train.py --num_episode 5000 --data_gen 1 --filename data_cube_5

# Collect grasp data
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so python train.py --grasp_total_steps 10 --filename data_cube_5_10_07_19_1612 --grasp_filename data_cube_5_10_07_19_1612_grasp --grasp_validation 1 --data_gen 1

# Training
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so python train.py --grasp_validation 1 --filename data_cube_5_10_07_19_1612 --trained_model data_cube_5_trained_model --num_episode 5000
'''
if __name__ ==  "__main__":
    # testing #
    # Initialize expert replay buffer, then generate expert pid data to fill it
    pid_mode = "naive"
    replay_size = 10
    with_grasp = False
    expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim=82, action_dim=3, max_episode=replay_size)
    replay_buffer, save_filepath, expert_saving_dir, text, num_success, total = GenerateExpertPID_JointVel(episode_num=10, requested_shapes=["CubeS"], requested_orientation="normal", with_grasp=with_grasp, replay_buffer=expert_replay_buffer, save=False, render_imgs=True, pid_mode=pid_mode)

    #print (replay_buffer, save_filepath)
    # plot_timestep_distribution(success_timesteps=None, fail_timesteps=None, all_timesteps=None, expert_saving_dir="12_8_expert_test_3x_100ts")
