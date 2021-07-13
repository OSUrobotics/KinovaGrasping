#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:14:16 2021

@author: orochi
"""
import numpy as np
from state_space import *
import DDPGfD
import os


class Controller():
    def __init__(self, controller_type, action_space, state_path=os.path.dirname(__file__)+'/config/controller_state.json'):
        self.state = StateSpaceBase(state_path)
        self.const_velocities = {"constant_velocity": 0.5, "min_velocity": 0.3, "max_velocity": 0.8, "finger_lift_velocity": 0.5, "wrist_lift_velocity": 0.6}
        self.lift_check = False
        self.timestep = 0
        self.state.update()
        self.PID = PID(action_space)
        self.init_obj_pose = self.state.get_value('Position_Obj')[0]#states[21]  # X position of object
        self.init_dot_prod = self.state.get_value('DotProduct_All')[-1]#states[81]  # dot product of object wrt palm
        if type(controller_type) == str:
            if controller_type == "naive":
                self.select_action = self.NaiveController
            elif controller_type == "bell-shaped":
                self.select_action = self.BellShapedController
            elif controller_type == "position-dependent":
                self.select_action = self.PDController
            elif controller_type == "combined":
                self.select_action = self.CombinedController
        else:
            self.select_action = controller_type.select_action

    def _set_lift_check(self, lift_check):
        self.lift_check = lift_check

    def CombinedController(self):
        """ Get action based on controller (Naive, position-dependent, combined interpolation)
            obs: Current state observation
            controller: Initialized expert PID controller
            env: Current Mujoco environment needed for expert PID controller
            return action: np.array([wrist, f1, f2, f3]) (velocities in rad/sec)
        """
        self.state.update()
        object_x_coord = self.state.get_value('Position_Obj')[0]  # Object x coordinate position
    
        # By default, action is set to close fingers at a constant velocity
        controller_action = np.array([self.const_velocities["constant_velocity"], self.const_velocities["constant_velocity"], self.const_velocities["constant_velocity"]])

        # If object x position is on outer edges, do expert pid
        if object_x_coord < -0.04 or object_x_coord > 0.04:
            # Expert Nudge controller strategy
            controller_action, f1_vels, f2_vels, f3_vels, wrist_vels = self.PDControl.PDController()

        # Object x position within the side-middle ranges, interpolate expert/naive velocity output
        elif -0.04 <= object_x_coord <= -0.02 or 0.02 <= object_x_coord <= 0.04:
            # Interpolate between naive and expert velocities
            # position-dependent controller action (finger velocity based on object location within hand)
            expert_action, f1_vels, f2_vels, f3_vels, wrist_vels = self.PDControl.PDController()
            # Naive controller action (fingers move at constant velocity)
            naive_action = self.NaiveController()

            # Interpolate finger velocity values between position-dependent and Naive action output
            finger_vels = np.interp(np.arange(0, 3), naive_action, expert_action)

            controller_action = np.array([finger_vels[0], finger_vels[1], finger_vels[2]])

        # Object x position is within center area, so use naive controller
        else:
            # Naive controller action (fingers move at constant velocity)
            controller_action = self.NaiveController()
    
        #print("**** action: ",action)
    
        return controller_action

    def NaiveController(self):
        """ Move fingers at a constant speed, return action """
    
        # By default, close all fingers at a constant speed
        action = np.array([self.const_velocities["constant_velocity"], self.const_velocities["constant_velocity"], self.const_velocities["constant_velocity"]])
    
        # If ready to lift, set fingers to constant lifting velocities
        if self.lift_check is True:
            action = np.array([self.const_velocities["finger_lift_velocity"], self.const_velocities["finger_lift_velocity"],
                               self.const_velocities["finger_lift_velocity"]])
    
        return action
    
    def BellShapedController(self):
        """ Move fingers at a constant speed, return action """
    
        bell_curve_velocities = [0.202, 0.27864, 0.35046, 0.41696, 0.47814, 0.534, 0.58454, 0.62976, 0.66966, 0.70424, 0.7335, 0.75744, 0.77606, 0.78936, 0.79734, 0.8, 0.79734, 0.78936, 0.77606, 0.75744, 0.7335, 0.70424, 0.66966, 0.62976, 0.58454, 0.534, 0.47814, 0.41696, 0.35046, 0.27864, 0.2015]
    
        # Determine the finger velocities by increasing and decreasing the values with a constant acceleration
        finger_velocity = bell_curve_velocities[self.timestep]
        self.timestep += 1
        # By default, close all fingers at a constant speed
        action = np.array([finger_velocity, finger_velocity, finger_velocity])
    
        # If ready to lift, set fingers to constant lifting velocities
        if self.lift_check is True:
            action = np.array([self.const_velocities["finger_lift_velocity"], self.const_velocities["finger_lift_velocity"],
                               self.const_velocities["finger_lift_velocity"]])
        #print("TS: {} action: {}".format(timestep, action))
    
        return action
   

    def center_action(self):
        """ Object is in a center location within the hand, so lift with constant velocity or adjust for lifting """
        obj_dot_prod = self.state.get_value('DotProduct_All')[-1]
        f1, f2, f3 = self.const_velocities['constant_velocity'], self.const_velocities['constant_velocity'], self.const_velocities['constant_velocity']

        if abs(obj_dot_prod - self.init_dot_prod) > 0.01:

            f1, f2, f3 = self.const_velocities['constant_velocity'], (self.const_velocities['constant_velocity'] / 2), (self.const_velocities['constant_velocity'] / 2)
        action = np.array([f1, f2, f3])
        if self.lift_check is True:
            action = np.array([self.const_velocities["finger_lift_velocity"], self.const_velocities["finger_lift_velocity"],
                               self.const_velocities["finger_lift_velocity"]])
        return action

    def right_action(self):
        """ Object is in an extreme right-side location within the hand, so Finger 2 and 3 move the
        object closer to the center """
        dot_prods = self.state.get_value('DotProduct_All')
        if abs(dot_prods[-1] - self.init_dot_prod) < 0.01:
            """ PRE-contact """
            f1 = 0.0  # frontal finger doesn't move
            f2 = self.PID.touch_vel(dot_prods[-1], dot_prods[4])  # f2_dist dot product to object
            f3 = f2  # other double side finger moves at same speed
        else:
            """ POST-contact """
            if abs(1 - dot_prods[-1]) > 0.01:
                f1 = self.const_velocities['min_velocity']  # frontal finger moves slightly
                f2 = self.PID.velocity(dot_prods[-1])  # get PID velocity
                f3 = f2  # other double side finger moves at same speed
            else:  # goal is within 0.01 of being reached:
                f1 = self.PID.touch_vel(dot_prods[-1], dot_prods[3])  # f1_dist dot product to object
                f2 = 0.0
                f3 = 0.0
        action = np.array([f1, f2, f3])
        if self.lift_check is True:
            action = np.array([self.const_velocities["finger_lift_velocity"], self.const_velocities["finger_lift_velocity"],
                               self.const_velocities["finger_lift_velocity"]])
        return action

    def left_action(self):
        """ Object is in an extreme left-side location within the hand, so Finger 1 moves the
                object closer to the center """
        dot_prods = self.state.get_value('DotProduct_All')
        # Only Small change in object dot prod to wrist from initial position, must move more
        if abs(dot_prods[-1] - self.init_dot_prod) < 0.01:
            """ PRE-contact """
            f1 = self.PID.touch_vel(dot_prods[-1], dot_prods[3])  # f1_dist dot product to object
            f2 = 0.0
            f3 = 0.0
        else:
            """ POST-contact """
            # now finger-object distance has been changed a decent amount.
            # Goal is 1 b/c obj_dot_prod is based on comparison of two normalized vectors
            if abs(1 - dot_prods[-1]) > 0.01:
                f1 = self.PID.velocity(dot_prods[-1])
                f2 = self.const_velocities['min_velocity']  # 0.05
                f3 = self.const_velocities['min_velocity']  # 0.05
            else:
                f2 = self.PID.touch_vel(dot_prods[-1], dot_prods[4])  # f2_dist dot product to object
                f3 = f2
                f1 = 0.0
        action = np.array([f1, f2, f3])
        if self.lift_check is True:
            action = np.array([self.const_velocities["finger_lift_velocity"], self.const_velocities["finger_lift_velocity"],
                               self.const_velocities["finger_lift_velocity"]])
        return action

    def PDController(self):
        """ Position-Dependent (PD) Controller that is dependent on the x-axis coordinate position of the object to 
        determine the individual finger velocities.
        """
        self.state.update()
        if abs(self.init_obj_pose) <= 0.03:
            controller_action = self.center_action()
        else:
            if self.init_obj_pose > 0.0:
                controller_action = self.right_action()

            else:
                controller_action = self.left_action()

        controller_action = self.check_vel_in_range(controller_action)

        return controller_action
    
    def check_vel_in_range(self, action):
        """ Checks that each of the finger/wrist velocies values are in range of min/max values """
        for idx in range(len(action)):
            if idx > 0:
                if action[idx] < self.const_velocities['min_velocity']:
                    if action[idx] != 0 or action[idx] != self.const_velocities['finger_lift_velocity'] or action[idx] != self.const_velocities['finger_lift_velocity'] / 2:
                        action[idx] = self.const_velocities['min_velocity']
                elif action[idx] > self.const_velocities['max_velocity']:
                    action[idx] = self.const_velocities['max_velocity']
    
        return action

class PID(object):
    generic_state_path = 'controller_state.json'
    def __init__(self):
        self.kp = 1
        self.kd = 1
        self.ki = 1
        self.prev_err = 0.0
        self.sampling_time = 15

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
        action = (vel) #* 0.3
        # if action < 0.8:
        #    action = 0.8
        #if action < 0.05:  # Old velocity
        #    action = 0.05
        return action
