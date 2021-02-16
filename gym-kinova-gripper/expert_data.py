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

# Import plotting code from other directory
plot_path = os.getcwd() + "/plotting_code"
sys.path.insert(1, plot_path)
from heatmap_coords import add_heatmap_coords, filter_heatmap_coords, coords_dict_to_array, save_coordinates

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

    def velocity(self, dot_prod):
        err = 1 - dot_prod
        diff = (err) / self.sampling_time
        vel = err * self.kp + diff * self.kd
        action = (vel / 1.25) * 0.3  # 1.25 means dot product equals to 1
        if action < 0.05:
            action = 0.05
        return action

    def joint(self, dot_prod):
        err = 1 - dot_prod
        diff = (err) / self.sampling_time
        joint = err * self.kp + diff * self.kd
        action = (joint / 1.25) * 2  # 1.25 means dot product equals to 1
        return action

    def touch_vel(self, obj_dotprod, finger_dotprod):
        err = obj_dotprod - finger_dotprod
        diff = err / self.sampling_time
        vel = err * self.kp + diff * self.kd
        action = (vel) * 0.3
        # if action < 0.8:
        #    action = 0.8
        if action < 0.05:  # Old velocity
            action = 0.05
        #print("TOUCH VEL, action: ", action)
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

        # Change in finger 2 distal x-coordinate position
        f2_change = abs(f_dist_old[3] - f_dist_new[3])
        f2_diff = f2_change / self.sampling_time

        # Change in finger 3 distal x-coordinate position
        f3_change = abs(f_dist_old[6] - f_dist_new[6])
        f3_diff = f3_change / self.sampling_time

        # Sum of changes in distal fingers
        f_all_change = f1_diff + f2_diff + f3_diff

        #print("f_all_change: ", f_all_change)

        # If the fingers have only changed a small amount, we assume the object is grasped
        if f_all_change < 0.0002:  # 0.0005: #0.00005
            return True
        else:
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
        # self.init_obj_pose = self._sim.data.get_geom_xpos(object)
        self.init_dot_prod = states[81]  # dot product of object wrt palm
        self.f1_vels = []
        self.f2_vels = []
        self.f3_vels = []
        self.wrist_vels = []

    def _count(self):
        self.step += 1

    def NudgeController(self, prev_states, states, action_space, constant_velocity, min_velocity, max_velocity, finger_lift_velocity, wrist_lift_velocity):
        # Define pid controller
        pid = PID(action_space)

        # Dot product of object wrt palm
        obj_dot_prod = states[81]

        # Define target region
        x = 0.0
        y = -0.01

        # Define finger action
        f1 = 0.0  # far out finger, on single side
        f2 = 0.0  # double side finger - right of finger 1
        f3 = 0.0  # double side finger - left of finger 1
        wrist = 0.0

        # Distal finger x,y,z positions f1_dist, f2_dist, f3_dist
        if prev_states is None:  # None if on the first timestep
            f_dist_old = None
        else:
            f_dist_old = prev_states[9:17]
        f_dist_new = states[9:17]

        ready_for_lift = 0
        num_consistent_grasps = 1  # Number of time steps the hand has to be in a good grasping position to lift

        ''' Note: only comparing initial X position of object. because we know
        the hand starts at the same position every time (close to origin) '''

        # Check if the object is near the center area (less than x-axis 0.03)
        if abs(self.init_obj_pose) <= 0.03:
            # print("CHECK 1: Object is near the center")
            f1, f2, f3 = constant_velocity, constant_velocity, constant_velocity

            # Check if change in object dot product to wrist center versus the initial dot product is greater than 0.01
            if abs(obj_dot_prod - self.init_dot_prod) > 0.01:
                # print("CHECK 2: Obj dot product to wrist has changed more than 0.01")
                # start lowering velocity of the three fingers
                f1, f2, f3 = constant_velocity, (constant_velocity / 2), (constant_velocity / 2)

                # Check if object is grasped - distal finger distance hasn't moved
                if pid.check_grasp(f_dist_old, f_dist_new) is True:
                    # print("Check 2A: Object is grasped, ready for lift")
                    if ready_for_lift >= num_consistent_grasps:
                        #print("Ready for lift!!!: ", ready_for_lift)
                        wrist, f1, f2, f3 = wrist_lift_velocity, (
                                    finger_lift_velocity / 2), finger_lift_velocity, finger_lift_velocity
                    ready_for_lift += 1
        else:
            # print("CHECK 3: Object is on extreme left OR right sides")
            # object on right hand side, move 2-fingered side
            # Local representation: POS X --> object is on the RIGHT (two fingered) side of hand
            if self.init_obj_pose > 0.0:
                # print("CHECK 4: Object is on RIGHT side")
                # Only Small change in object dot prod to wrist from initial position, must move more
                if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
                    """ PRE-contact """
                    # print("CHECK 5: Only Small change in object dot prod to wrist, moving f2 & f3")
                    f1 = 0.0  # frontal finger doesn't move
                    f2 = pid.touch_vel(obj_dot_prod, states[79])  # f2_dist dot product to object
                    f3 = f2  # other double side finger moves at same speed
                    wrist = 0.0
                else:
                    """ POST-contact """
                    # now finger-object distance has been changed a decent amount.
                    # print("CHECK 6: Object dot prod to wrist has Changed More than 0.01")
                    # Goal is 1 b/c obj_dot_prod is based on comparison of two normalized vectors
                    if abs(1 - obj_dot_prod) > 0.01:
                        # print("CHECK 7: Obj dot prod to wrist is > 0.01, so moving ALL f1, f2 & f3")
                        # start to close the PID stuff
                        f1 = min_velocity # frontal finger moves slightly
                        f2 = pid.velocity(obj_dot_prod)  # get PID velocity
                        f3 = f2  # other double side finger moves at same speed
                        wrist = 0.0
                    else:  # goal is within 0.01 of being reached:
                        # print("CHECK 8: Obj dot prod to wrist is Within reach of 0.01 or less, Move F1 Only")
                        # start to close from the first finger
                        f1 = pid.touch_vel(obj_dot_prod, states[78])  # f1_dist dot product to object
                        f2 = 0.0
                        f3 = 0.0
                        wrist = 0.0

                    # print("Check 9a: Check for grasp (small distal finger movement)")
                    # Check for a good grasp (small distal finger movement)
                    if pid.check_grasp(f_dist_old, f_dist_new) is True:
                        # print("CHECK 9: Yes! Good grasp, move ALL fingers")
                        if ready_for_lift >= num_consistent_grasps:
                            #print("Ready for lift!!!: ", ready_for_lift)
                            wrist, f1, f2, f3 = wrist_lift_velocity, (
                                        finger_lift_velocity / 2), finger_lift_velocity, finger_lift_velocity
                        ready_for_lift += 1

            # object on left hand side, move 1-fingered side
            # Local representation: NEG X --> object is on the LEFT (thumb) side of hand
            else:
                # print("CHECK 10: Object is on the LEFT side")
                # Only Small change in object dot prod to wrist from initial position, must move more
                if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
                    """ PRE-contact """
                    # print("CHECK 11: Only Small change in object dot prod to wrist, moving F1")
                    f1 = pid.touch_vel(obj_dot_prod, states[78])  # f1_dist dot product to object
                    f2 = 0.0
                    f3 = 0.0
                    wrist = 0.0
                else:
                    """ POST-contact """
                    # now finger-object distance has been changed a decent amount.
                    # print("CHECK 12: Object dot prod to wrist has Changed More than 0.01")
                    # Goal is 1 b/c obj_dot_prod is based on comparison of two normalized vectors
                    if abs(1 - obj_dot_prod) > 0.01:
                        # print("CHECK 13: Obj dot prod to wrist is > 0.01, so kep moving f1, f2 & f3")
                        f1 = pid.velocity(obj_dot_prod)
                        f2 = min_velocity  # 0.05
                        f3 = min_velocity  # 0.05
                        wrist = 0.0
                    else:
                        # Goal is within 0.01 of being reached:
                        # print("CHECK 14: Obj dot prod to wrist is Within reach of 0.01 or less, Move F2 & F3 Only")
                        # start to close from the first finger
                        # nudge with thumb
                        f2 = pid.touch_vel(obj_dot_prod, states[79])  # f2_dist dot product to object
                        f3 = f2
                        f1 = 0.0
                        wrist = 0.0

                    # print("Check 15a: Check for grasp (small distal finger movement)")
                    # Check for a good grasp (small distal finger movement)
                    if pid.check_grasp(f_dist_old, f_dist_new) is True:
                        # print("CHECK 15: Good grasp - moving ALL fingers")
                        # wrist, f1, f2, f3 = 0.6, 0.15, 0.3, 0.3
                        if ready_for_lift >= num_consistent_grasps:
                            #print("Ready for lift!!!: ", ready_for_lift)
                            wrist, f1, f2, f3 = wrist_lift_velocity, (
                                        finger_lift_velocity / 2), finger_lift_velocity, finger_lift_velocity
                            # ready_for_lift = 0
                        ready_for_lift += 1

        self._count()
        hand_velocities = np.array([wrist, f1, f2, f3])
        hand_velocities = check_vel_in_range(hand_velocities, min_velocity, max_velocity, finger_lift_velocity)

        #print("f1: ", hand_velocities[1], " f2: ", hand_velocities[2], " f3: ", hand_velocities[3], " wrist: ",
        #      hand_velocities[0])
        self.f1_vels.append(f1)
        self.f2_vels.append(f2)
        self.f3_vels.append(f3)
        self.wrist_vels.append(f3)

        return hand_velocities, ready_for_lift, self.f1_vels, self.f2_vels, self.f3_vels, self.wrist_vels


def check_vel_in_range(hand_velocities, min_velocity, max_velocity, finger_lift_velocity):
    """ Checks that each of the finger/wrist velocies values are in range of min/max values """
    for idx in range(len(hand_velocities)):
        if idx > 0:
            if hand_velocities[idx] < min_velocity:
                if hand_velocities[idx] != 0 or hand_velocities[idx] != finger_lift_velocity or hand_velocities[idx] != finger_lift_velocity / 2:
                    hand_velocities[idx] = min_velocity
            elif hand_velocities[idx] > max_velocity:
                hand_velocities[idx] = max_velocity

    return hand_velocities


def GenerateTestPID_JointVel(obs, env):
    controller = ExpertPIDController(obs)
    action = controller.NudgeController(obs, env.action_space)
    return action


def naive_check_grasp(f_dist_old, f_dist_new):
    """
    Uses the current change in x,y position of the distal finger tips, summed over all fingers to determine if
    the object is grasped (fingers must have only changed in position over a tiny amount to be considered done).
    @param: f_dist_old: Distal finger tip x,y,z coordinate values from previous timestep
    @param: f_dist_new: Distal finger tip x,y,z coordinate values from current timestep
    """

    # Initial check to see if previous state has been set
    if f_dist_old is None:
        return [0,0]
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


def get_action(prev_obs, obs, total_steps, controller, env, f_dist_old, f_dist_new, ready_for_lift,
               num_consistent_grasps, pid_mode="expert_naive"):
    """ Get action based on position of object within the hand
        @param prev_obs:
        @param obs:
        @param total_steps:
        @param controller:
        @param env:
        @param f_dist_old:
        @param f_dist_new:
        @param ready_for_lift:
        @param num_consistent_grasps:
        return action: np.array([wrist, f1, f2, f3])
    """
    constant_velocity = 0.5
    min_velocity = 0.5
    max_velocity = 0.8
    finger_lift_velocity = min_velocity
    wrist_lift_velocity = 0.6
    min_lift_timesteps = 10
    object_x_coord = obs[21]  # Object x coordinate position
    # Action is Naive Controller by default
    action = np.array([0, constant_velocity, constant_velocity, constant_velocity])

    # Check for Naive Only or Expert Only setting
    # Only Naive constant velocity
    if pid_mode == "naive_only":
        naive_ret = naive_check_grasp(f_dist_old, f_dist_new)
        if bool(naive_ret[0]) is True and total_steps > min_lift_timesteps:
            if ready_for_lift >= num_consistent_grasps:
                print("Ready for lift!!!: ", ready_for_lift)
                action = np.array([wrist_lift_velocity, finger_lift_velocity, finger_lift_velocity,
                                   finger_lift_velocity])
            ready_for_lift += 1
            # Checks the grasp is good over num_consistent_grasps number of trials
            if ready_for_lift >= num_consistent_grasps:
                action[0] = wrist_lift_velocity
        else:
            action = np.array([0, constant_velocity, constant_velocity, constant_velocity])

        return action

    # Only Expert nudge strategy
    elif pid_mode == "expert_only":
        # Expert Nudge controller strategy
        action, ready_for_lift, f1_vels, f2_vels, f3_vels, wrist_vels = controller.NudgeController(
            prev_obs, obs, env.action_space, constant_velocity, min_velocity, max_velocity,
            finger_lift_velocity, wrist_lift_velocity)
        # Do not lift until after min time steps
        if total_steps < min_lift_timesteps:
            action[0] = 0
        # Checks the grasp is good over num_consistent_grasps number of trials
        elif ready_for_lift >= num_consistent_grasps:
            action[0] = wrist_lift_velocity
        return action

    # Interpolate Naive and Expert Strategies based on object x-coord position within hand
    else:
        # If object x position is on outer edges, do expert pid
        if object_x_coord < -0.04 or object_x_coord > 0.04:
            # Expert Nudge controller strategy
            action, ready_for_lift, f1_vels, f2_vels, f3_vels, wrist_vels = controller.NudgeController(
                prev_obs, obs, env.action_space, constant_velocity, min_velocity, max_velocity,
                finger_lift_velocity, wrist_lift_velocity)
            # Do not lift until after 50 steps
            if total_steps < min_lift_timesteps:
                action[0] = 0
        # Object x position within the side-middle ranges, interpolate expert/naive velocity output
        elif -0.04 <= object_x_coord <= -0.02 or 0.02 <= object_x_coord <= 0.04:
            # Interpolate between naive and expert velocities
            # Default Naive Controller velocities
            naive_action = np.array([0, constant_velocity, constant_velocity, constant_velocity])

            naive_ret = naive_check_grasp(f_dist_old, f_dist_new)
            if bool(naive_ret[0]) is True and total_steps > min_lift_timesteps:
                if ready_for_lift >= num_consistent_grasps:
                    naive_action = np.array([wrist_lift_velocity, finger_lift_velocity, finger_lift_velocity,
                                             finger_lift_velocity])
                ready_for_lift += 1

            # Expert PID
            expert_action, ready_for_lift, f1_vels, f2_vels, f3_vels, wrist_vels = controller.NudgeController(
                prev_obs, obs, env.action_space, constant_velocity, min_velocity, max_velocity,
                finger_lift_velocity, wrist_lift_velocity)

            if naive_action[0] == 0:
                wrist_vel = 0
            else:
                wrist_vel = naive_action[0] + expert_action[0] / 2

            finger_vels = np.interp(np.arange(1, 4), naive_action[1:3], expert_action[1:3])

            # Only start to lift if we've had some RL time steps (multiple actions) to adjust hand
            if total_steps < min_lift_timesteps:
                wrist_vel = 0

            action = np.array([wrist_vel, finger_vels[0], finger_vels[1], finger_vels[2]])

        # Object x position is within center area, so use naive controller
        else:
            naive_ret = naive_check_grasp(f_dist_old, f_dist_new)
            if bool(naive_ret[0]) is True and total_steps > min_lift_timesteps:
                if ready_for_lift >= num_consistent_grasps:
                    action = np.array([wrist_lift_velocity, finger_lift_velocity, finger_lift_velocity,
                                       finger_lift_velocity])
                ready_for_lift += 1
            else:
                action = np.array([0, constant_velocity, constant_velocity, constant_velocity])

    # Checks the grasp is good over num_consistent_grasps number of trials
    if ready_for_lift >= num_consistent_grasps:
        action[0] = wrist_lift_velocity

    return action


def GenerateExpertPID_JointVel(episode_num, requested_shapes, requested_orientation, with_grasp, replay_buffer=None, save=True, render_imgs=False, pid_mode="expert_naive"):
    """ Generate expert data based on Expert PID and Naive PID controller action output.
    @param episode_num: Number of episodes to generate expert data for
    @param replay_buffer: Replay buffer to be passed in (set to None for testing purposes)
    @param save: set to True if replay buffer data is to be saved
    """
    env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    env.env.pid = True

    success_coords = {"x": [], "y": [], "orientation": []}
    fail_coords = {"x": [], "y": [], "orientation": []}
    # hand orientation types: NORMAL, Rotated (45 deg), Top (90 deg)

    all_timesteps = np.array([])  # Keeps track of all timesteps to determine average timesteps needed
    success_timesteps = np.array([])  # Successful episode timesteps count distribution
    fail_timesteps = np.array([])  # Failed episode timesteps count distribution
    datestr = datetime.datetime.now().strftime("%m_%d_%y_%H%M") # used for file name saving

    print("----Generating {} expert episodes----".format(episode_num))
    print("Using PID MODE: ",pid_mode)

    # Beginning of episode loop
    for i in range(episode_num):
        print("PID ", i)
        # Fill training object list using latin square
        if env.check_obj_file_empty("objects.csv") or episode_num is 0:
            env.Generate_Latin_Square(episode_num, "objects.csv", shape_keys=requested_shapes)
        obs, done = env.reset(shape_keys=requested_shapes,hand_orientation=requested_orientation), False

        prev_obs = None         # State observation of the previous state
        ready_for_lift = 0      # Signals if ready for lift, from check_grasp()
        num_consistent_grasps = 1   # Number of RL steps needed
        total_steps = 0             # Total RL timesteps passed within episode
        env._max_episode_steps = 30 # Sets number of timesteps per episode (counted from each step() call)
        obj_coords = env.get_obj_coords()
        # Local coordinate conversion
        obj_local = np.append(obj_coords,1)
        obj_local = np.matmul(env.Tfw,obj_local)
        obj_local_pos = obj_local[0:3]

        action_str = None
        controller = ExpertPIDController(obs)   # Initiate expert PID controller

        # Record episode starting index if replay buffer is passed in
        if replay_buffer is not None:
            replay_buffer.add_episode(1)

        # Beginning of RL time steps within the current episode
        while not done:
            # Render image from current episode
            if render_imgs is True:
                if total_steps % 1 == 0:
                    env.render_img(dir_name=pid_mode+"_"+datestr,text_overlay=action_str, episode_num=i, timestep_num=total_steps,
                                   obj_coords=str(obj_local_pos[0]) + "_" + str(obj_local_pos[1]))
                else:
                    env._viewer = None

            # Distal finger x,y,z positions f1_dist, f2_dist, f3_dist
            if prev_obs is None:  # None if on the first RL episode time step
                f_dist_old = None
            else:
                f_dist_old = prev_obs[9:17]
            f_dist_new = obs[9:17]

            # Get action based on the location of the object within the hand
            action = get_action(prev_obs, obs, total_steps, controller, env, f_dist_old, f_dist_new,
                                ready_for_lift, num_consistent_grasps, pid_mode)

            # Used for the action string
            ###After taking step###
            [naive_ret,_] = naive_check_grasp(f_dist_old, f_dist_new)

            # Take action (Reinforcement Learning step)
            env.set_with_grasp_reward(with_grasp)
            next_obs, reward, done, info = env.step(action)

            # Add experience to replay buffer
            if replay_buffer is not None and not naive_ret:
                replay_buffer.add(obs[0:82], action, next_obs[0:82], reward, float(done))

            if naive_ret and done:
                replay_buffer.replace(reward, done)
                # print ("#######REWARD#######", reward)
                # replay_buffer.add(obs[0:82], action, next_obs[0:82], reward, float(done))

            # if replay_buffer is not None and not naive_ret and not done:
            #     replay_buffer.add(obs[0:82], action, next_obs[0:82], reward, float(done))
            #
            # if done:
            #     # print ("#######REWARD#######", reward)
            #     # replay_buffer.add(obs[0:82], action, next_obs[0:82], reward, float(done))
            #     replay_buffer.replace(reward, done)

            action_str = "Wrist: " + str(action[0]) + "\nFinger1: " + str(action[1]) + "\nFinger2: " + str(
                action[2]) + "\nFinger3: " + str(action[3]) + "\nready_for_lift: " + str(
                ready_for_lift) + "\nf_all_change: " + str(naive_ret) + "\nobject center height: " + str(obs[23]) +\
                         "\ntimestep reward: " + str(reward) +"\nfinger reward: " + str(info["finger_reward"]) +\
                         "\ngrasp reward: " + str(info["grasp_reward"]) + "\nlift reward: " + str(info["lift_reward"])

            # Once current timestep is over, update prev_obs to be current obs
            if total_steps > 0:
                prev_obs = obs
            obs = next_obs
            total_steps += 1

        all_timesteps = np.append(all_timesteps, total_steps)

        # print("Expert PID total timestep: ", total_steps)
        lift_success = None
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
            if total_steps % 1 == 0:
                env.render_img(dir_name=pid_mode+"_"+datestr,text_overlay=str(action), episode_num=i, timestep_num=total_steps,
                               obj_coords=str(obj_local_pos[0]) + "_" + str(obj_local_pos[1]), final_episode_type=lift_success)

        # Add heatmap coordinates
        orientation = env.get_orientation()
        ret = add_heatmap_coords(success_coords, fail_coords, orientation, obj_local_pos, success)
        success_coords = copy.deepcopy(ret["success_coords"])
        fail_coords = copy.deepcopy(ret["fail_coords"])

        if replay_buffer is not None:
            replay_buffer.add_episode(0)

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
    expert_saving_dir = "./expert_replay_data/"+grasp_str+"/"+str(pid_mode)+"/"+str(shapes_str)
    expert_output_saving_dir = expert_saving_dir + "/output"
    expert_replay_saving_dir = expert_saving_dir + "/replay_buffer"
    heatmap_saving_dir = expert_output_saving_dir + "/heatmap/expert"
    heatmap_save_path = Path(heatmap_saving_dir)
    heatmap_save_path.mkdir(parents=True, exist_ok=True)

    # Filter heatmap coords by success/fail, orientation type, and save to appropriate place
    filter_heatmap_coords(success_coords, fail_coords, None, heatmap_saving_dir)

    #print("Plotting timestep distribution...")
    #plot_timestep_distribution(success_timesteps, fail_timesteps, all_timesteps, expert_saving_dir)

    #print("Save is: ", str(save))
    save_filepath = None
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

        print("Expert data generation replay buffer saved to location: ", save_filepath)
        print("# Episodes: ", replay_buffer.replay_ep_num)
        print("# Trajectories: ", replay_buffer.size)
        print("\nHeatmap coordinate data saved at: ",expert_saving_dir)
    return replay_buffer, save_filepath, expert_saving_dir, text, num_success, (num_success+num_fail)


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
if __name__ ==  "__main__":
    # testing #
    # Initialize expert replay buffer, then generate expert pid data to fill it
    state_dim = 82
    action_dim = 4
    expert_replay_size = 10
    with_grasp = False
    expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
    replay_buffer, save_filepath, data_dir, info_file_text = GenerateExpertPID_JointVel(10, "CubeS", with_grasp, expert_replay_buffer, save=False, render_imgs=True, pid_mode="expert_naive")

    #print (replay_buffer, save_filepath)
    # plot_timestep_distribution(success_timesteps=None, fail_timesteps=None, all_timesteps=None, expert_saving_dir="12_8_expert_test_3x_100ts")
