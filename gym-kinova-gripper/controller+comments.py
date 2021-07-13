#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:14:16 2021

@author: orochi
"""
import numpy as np
from state_space import *
import DDPGfD

#            controller_action, f1_vels, f2_vels, f3_vels, wrist_vels = controller.PDController(lift_check, obs, env.action_space, velocities)
    
class Controller():
    def __init__(self, controller_type, state_path='controller_state.json'):
        self.state = StateSpace(state_path)
        self.const_velocities = {"constant_velocity": 0.5, "min_velocity": 0.3, "max_velocity": 0.8, "finger_lift_velocity": 0.5, "wrist_lift_velocity": 0.6}
        self.lift_check = False
        self.timestep = 0
        self.prev_f1jA = 0.0
        self.prev_f2jA = 0.0
        self.prev_f3jA = 0.0
        self.step = 0.0
        self.state.update()
        self.state.get_full_arr()
        self.PID = PID(action_space)
        self.init_obj_pose = self.state.get_value('Obj')[0]#states[21]  # X position of object
        self.init_dot_prod = self.state.get_value(['FingerObjDist','DotProduct_All'])[-1]#states[81]  # dot product of object wrt palm
        if type(controller_type) == str:
            if controller_type == "naive":
                self.select_action = self.NaiveController
            elif controller_type == "bell-shaped":
                self.select_action = self.BellShapedController
            elif controller_type == "position-dependent":
                self.PDControl = PositionDependentControl(self.state)
                self.select_action = PDControl.PDController
            elif controller_type == "combined":
                self.select_action = self.CombinedController
        self.f1_vels = []
        self.f2_vels = []
        self.f3_vels = []
        self.wrist_vels = []

    def _count(self):
        self.step += 1
  
    def CombinedController(self, lift_check, controller, env, pid_mode="combined",timestep=None):
        """ Get action based on controller (Naive, position-dependent, combined interpolation)
            obs: Current state observation
            controller: Initialized expert PID controller
            env: Current Mujoco environment needed for expert PID controller
            return action: np.array([wrist, f1, f2, f3]) (velocities in rad/sec)
        """
        self.state.update()
        object_x_coord = self.state.get_value('Obj')[0]  # Object x coordinate position
    
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
#      
#    def get_action(obs, lift_check, controller, env, pid_mode="combined",timestep=None):
#        """ Get action based on controller (Naive, position-dependent, combined interpolation)
#            obs: Current state observation
#            controller: Initialized expert PID controller
#            env: Current Mujoco environment needed for expert PID controller
#            return action: np.array([wrist, f1, f2, f3]) (velocities in rad/sec)
#        """
#        velocities = {"constant_velocity": 0.5, "min_velocity": 0.3, "max_velocity": 0.8, "finger_lift_velocity": 0.5, "wrist_lift_velocity": 0.6}
#        object_x_coord = obs[21]  # Object x coordinate position
#    
#        # By default, action is set to close fingers at a constant velocity
#        controller_action = np.array([velocities["constant_velocity"], velocities["constant_velocity"], velocities["constant_velocity"]])
#    
#        # NAIVE CONTROLLER: Close all fingers at a constant speed
#        if pid_mode == "naive":
#            controller_action = NaiveController()
#    
#        # POSITION-DEPENDENT CONTROLLER: Only move fingers based on object x-coord position within hand
#        elif pid_mode == "position-dependent":
#            controller_action, f1_vels, f2_vels, f3_vels, wrist_vels = controller.PDController(lift_check, obs, env.action_space, velocities)
#    
#        elif pid_mode == "bell-shaped":
#            controller_action = BellShapedController(lift_check, velocities, timestep)
#    
#        # COMBINED CONTROLLER: Interpolate Naive and Position-Dependent controller output based on object x-coord position within hand
#        else:
#            # If object x position is on outer edges, do expert pid
#            if object_x_coord < -0.04 or object_x_coord > 0.04:
#                # Expert Nudge controller strategy
#                controller_action, f1_vels, f2_vels, f3_vels, wrist_vels = controller.PDController(lift_check, obs, env.action_space, velocities)
#    
#            # Object x position within the side-middle ranges, interpolate expert/naive velocity output
#            elif -0.04 <= object_x_coord <= -0.02 or 0.02 <= object_x_coord <= 0.04:
#                # Interpolate between naive and expert velocities
#                # position-dependent controller action (finger velocity based on object location within hand)
#                expert_action, f1_vels, f2_vels, f3_vels, wrist_vels = controller.PDController(lift_check, obs, env.action_space, velocities)
#                # Naive controller action (fingers move at constant velocity)
#                naive_action = NaiveController(lift_check, velocities)
#    
#                # Interpolate finger velocity values between position-dependent and Naive action output
#                finger_vels = np.interp(np.arange(0, 3), naive_action, expert_action)
#    
#                controller_action = np.array([finger_vels[0], finger_vels[1], finger_vels[2]])
#    
#            # Object x position is within center area, so use naive controller
#            else:
#                # Naive controller action (fingers move at constant velocity)
#                controller_action = NaiveController(lift_check, velocities)
#    
#        #print("**** action: ",action)
#    
#        return controller_action
#    
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
    
class PID(object):
    generic_state_path = 'controller_state.json'
    def __init__(self, state, action_space):
        self.kp = 1
        self.kd = 1
        self.ki = 1
        self.prev_err = 0.0
        self.sampling_time = 15
        self.action_range = [action_space.low, action_space.high]
        self.state = state

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

    def center_action(self, constant_velocity, wrist_lift_velocity, finger_lift_velocity, obj_dot_prod, lift_check):
        """ Object is in a center location within the hand, so lift with constant velocity or adjust for lifting """
        wrist, f1, f2, f3 = 0, constant_velocity, constant_velocity, constant_velocity

        # Check if change in object dot product to wrist center versus the initial dot product is greater than 0.01
        if abs(obj_dot_prod - self.init_dot_prod) > 0.01:
            #print("CHECK 2: Obj dot product to wrist has changed more than 0.01")
            # Start lowering velocity of finger 2 and 3 so the balance of force is equal (no tipping)
            f1, f2, f3 = constant_velocity, (constant_velocity / 2), (constant_velocity / 2)

        # Lift check determined by grasp check (distal finger tip movements)
        # and this check has occurred over multiple time steps
        if lift_check is True:
            # Ready to lift, so slow down Finger 1 to allow for desired grip
            # (where Fingers 2 and 3 have dominance)
            #print("Check 2A: Object is grasped, ready for lift")
            f1, f2, f3 = (finger_lift_velocity / 2), finger_lift_velocity, finger_lift_velocity
        return np.array([f1, f2, f3])

    def right_action(self, pid, states, min_velocity, wrist_lift_velocity, finger_lift_velocity, obj_dot_prod, lift_check):
        """ Object is in an extreme right-side location within the hand, so Finger 2 and 3 move the
        object closer to the center """
        # Only Small change in object dot prod to wrist from initial position, must move more
        # Object has not moved much, we want the fingers to move closer to the object to move it
        if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
            """ PRE-contact """
            #print("CHECK 5: Only Small change in object dot prod to wrist, moving f2 & f3")
            f1 = 0.0  # frontal finger doesn't move
            f2 = pid.touch_vel(obj_dot_prod, states[79])  # f2_dist dot product to object
            f3 = f2  # other double side finger moves at same speed
            wrist = 0.0
        else:
            """ POST-contact """
            # now finger-object distance has been changed a decent amount.
            #print("CHECK 6: Object dot prod to wrist has Changed More than 0.01")
            # Goal is 1 b/c obj_dot_prod is based on comparison of two normalized vectors
            if abs(1 - obj_dot_prod) > 0.01:
                #print("CHECK 7: Obj dot prod to wrist is > 0.01, so moving ALL f1, f2 & f3")
                # start to close the PID stuff
                f1 = min_velocity  # frontal finger moves slightly
                f2 = pid.velocity(obj_dot_prod)  # get PID velocity
                f3 = f2  # other double side finger moves at same speed
                wrist = 0.0
            else:  # goal is within 0.01 of being reached:
                #print("CHECK 8: Obj dot prod to wrist is Within reach of 0.01 or less, Move F1 Only")
                # start to close from the first finger
                f1 = pid.touch_vel(obj_dot_prod, states[78])  # f1_dist dot product to object
                f2 = 0.0
                f3 = 0.0
                wrist = 0.0

            #print("Check 9a: Check for grasp (small distal finger movement)")
            # Lift check determined by grasp check (distal finger tip movements)
            # and this check has occurred over multiple time steps
            if lift_check is True:
                #print("CHECK 9: Yes! Good grasp, move ALL fingers")
                f1, f2, f3 = (finger_lift_velocity / 2), finger_lift_velocity, finger_lift_velocity

        return np.array([f1, f2, f3])

    def left_action(self, pid, states, min_velocity, wrist_lift_velocity, finger_lift_velocity, obj_dot_prod, lift_check):
        """ Object is in an extreme left-side location within the hand, so Finger 1 moves the
                object closer to the center """
        # Only Small change in object dot prod to wrist from initial position, must move more
        if abs(obj_dot_prod - self.init_dot_prod) < 0.01:
            """ PRE-contact """
            #print("CHECK 11: Only Small change in object dot prod to wrist, moving F1")
            f1 = pid.touch_vel(obj_dot_prod, states[78])  # f1_dist dot product to object
            f2 = 0.0
            f3 = 0.0
            wrist = 0.0
        else:
            """ POST-contact """
            # now finger-object distance has been changed a decent amount.
            #print("CHECK 12: Object dot prod to wrist has Changed More than 0.01")
            # Goal is 1 b/c obj_dot_prod is based on comparison of two normalized vectors
            if abs(1 - obj_dot_prod) > 0.01:
                #print("CHECK 13: Obj dot prod to wrist is > 0.01, so kep moving f1, f2 & f3")
                f1 = pid.velocity(obj_dot_prod)
                f2 = min_velocity  # 0.05
                f3 = min_velocity  # 0.05
                wrist = 0.0
            else:
                # Goal is within 0.01 of being reached:
                #print("CHECK 14: Obj dot prod to wrist is Within reach of 0.01 or less, Move F2 & F3 Only")
                # start to close from the first finger
                # nudge with thumb
                f2 = pid.touch_vel(obj_dot_prod, states[79])  # f2_dist dot product to object
                f3 = f2
                f1 = 0.0
                wrist = 0.0

            #print("Check 15a: Check for grasp (small distal finger movement)")
            # Lift check determined by grasp check (distal finger tip movements)
            # and this check has occurred over multiple time steps
            if lift_check is True:
                #print("CHECK 15b: Good grasp - moving ALL fingers")
                f1, f2, f3 = (finger_lift_velocity / 2), finger_lift_velocity, finger_lift_velocity
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
        min_velocity = velocities["min_velocity"]
        max_velocity = velocities["max_velocity"]

        # Note: only comparing initial X position of object. because we know
        # the hand starts at the same position every time (close to origin)

        # Check if the object is near the center area (less than x-axis 0.03)
        if abs(self.init_obj_pose) <= 0.03:
            #print("CHECK 1: Object is near the center")
            controller_action = self.center_action(constant_velocity, wrist_lift_velocity, finger_lift_velocity, obj_dot_prod, lift_check)
        else:
            #print("CHECK 3: Object is on extreme left OR right sides")
            # Object on right hand side, move 2-fingered side
            # Local representation: POS X --> object is on the RIGHT (two fingered) side of hand
            if self.init_obj_pose > 0.0:
                #print("CHECK 4: Object is on RIGHT side")
                controller_action = self.right_action(pid, states, min_velocity, wrist_lift_velocity, finger_lift_velocity, obj_dot_prod, lift_check)

            # object on left hand side, move 1-fingered side
            # Local representation: NEG X --> object is on the LEFT (thumb) side of hand
            else:
                #print("CHECK 10: Object is on the LEFT side")
                controller_action = self.left_action(pid, states, min_velocity, wrist_lift_velocity, finger_lift_velocity, obj_dot_prod, lift_check)

        self._count()
        controller_action = check_vel_in_range(controller_action, min_velocity, max_velocity, finger_lift_velocity)

        #print("f1: ", controller_action[0], " f2: ", controller_action[1], " f3: ", controller_action[2])
        self.f1_vels.append(f1)
        self.f2_vels.append(f2)
        self.f3_vels.append(f3)
        self.wrist_vels.append(wrist)

        return controller_action, self.f1_vels, self.f2_vels, self.f3_vels, self.wrist_vels