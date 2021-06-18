""" Create a simple rendering of the hand and object """

import gym
import numpy as np

action = np.array([0.0, 0.5, 0.5, 0.5])
t = 0
requested_shapes = ["CubeM"]

env = gym.make('gym_kinova_gripper:kinovagripper-v0')
env.Generate_Latin_Square(100, "objects.csv", shape_keys=requested_shapes)
for i in range(100):
    env.reset(shape_keys=requested_shapes,hand_orientation="top")
    done = False
    while not done:
        obs, reward, done, _ = env.step(action)
        env.render()
        env.pause()
    env.env.close()
