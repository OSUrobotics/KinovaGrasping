#!/usr/bin/env python3
import gym
import time
import numpy as np


class CollisionCheck:
    """
    This function will check for any possible collisions in the starting coordinates of the data set
    """
    def __init__(self, data_points, orn_type=None):
        """
        :param data_points: Can be  array of dataset or list of single starting coordinates
        list will be in the order:
        [x, y, z, ornx, orny, ornz]
        array will be of new dataset order:
        [idx, Noise, Orientation, ornx, orny, ornz, shapesize, x, y, z, controllertype]
        or old dataset  order:
        [x, y, z, ornx, orny, ornz]
        """
        pass


def collision_check_new_coords(env, data_points):
    """
    Data  points will be  an single length array of the new dataset of order:
    [idx, Noise, Orientation, ornx, orny, ornz, shapesize, x, y, z, controllertype,  xml_file]
    :param data_points:
    :return: collision: True (There is a collision) / False (No Collision)
    """

    ornx = data_points[3]
    orny = data_points[4]
    ornz = data_points[5]
    objx = data_points[7]
    objy = data_points[8]
    objz = data_points[9]
    env.write_xml(np.array([ornx, orny, ornz]))
    env.place_obj(objx, objy, objz)


if __name__ == '__main__':
    env = gym.make("gym_kinova_gripper:kinovagripper-v0")

    # dataset_ex1 = [0, True, 'Normal', -1.57, 0.0, -1.57, 'CubeS', 0.03, 0.03, 0.0654, 'Policy', '/Users/asar/Desktop/Grimm\'s Lab/Grasping/Codes/KinovaGrasping/gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_CubeS.xml']
    # dataset_ex2 = [0, True, 'Normal', -1.57, 0.0, -1.57, 'CubeS', 0.08, 0.03, 0.0654, 'Policy', '/Users/asar/Desktop/Grimm\'s Lab/Grasping/Codes/KinovaGrasping/gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_CubeS.xml']
    dataset_ex1 = [0, True, 'Normal', -1.57, 0.0, -1.57, 'CubeS', 0.03, 0.03, 0.0654, 'Policy', '/Users/asar/Desktop/Grimm\'s Lab/Grasping/Codes/KinovaGrasping/gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1.xml']
    dataset_ex2 = [0, True, 'Normal', -1.57, 0.0, -1.57, 'CubeS', 0.08, 0.03, 0.0654, 'Policy', '/Users/asar/Desktop/Grimm\'s Lab/Grasping/Codes/KinovaGrasping/gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1.xml']

    env.reset()
    env.real_reset(dataset_ex1)
    env.is_it_still_there(0.0, 0.0, 0.065400)

    # while(1):
        # for i in range(0, 500):
    env.step([0.0,0.0,0.0,0.0])
    # env.real_step([0.0,0.0,0.0,0.0])
    env.is_it_still_there(0.0, 0.0, 0.065400)
    env.render()
    # env.reset()
    #
    # env.step([0.0,0.0,0.0,0.0])
    # env.render()
    env.reset()

    env.real_reset(dataset_ex2)
    env.is_it_still_there(0.08, 0.03, 0.0654)
    # for i in range(0, 500):
    env.step([0.0,0.0,0.0,0.0])
    # env.real_step([0.0,0.0,0.0,0.0])
    env.is_it_still_there(0.08, 0.03, 0.0654)
    env.render()
    # env.reset()
    #
    # env.step([0.0,0.0,0.0,0.0])
    # env.render()
    # env.reset()

    while(1):
        time.sleep(0.01)
