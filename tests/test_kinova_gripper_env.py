#!/usr/bin/env python3

from unittest import TestCase

import unittest
import gym
import numpy as np


class TestKinovaGripper_Env(TestCase):

    # Environment class initialization tests
    def test_env_instantiation(self):
        """ Test the environment class is instantiated correctly given the metadata produces by the Experiment Setup class"""
        # OpenAI gym environment name
        env_name = "gym_kinova_gripper:kinovagripper-v0"
        metadata = {"idx": 0, "Noise": True, "Orn": "normal", "Orn_Values": [0.01, 1.23, -3.4], "Shape": "CubeS",
                    "Start_Values": [1.1, -2.3, 0.0],
                    "xml_file": "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"}

        env = gym.make(env_name)  # Make initial environment

        env.reset(metadata=metadata)

        self.assertEqual(env.filename, "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml")
        self.assertEqual(env.random_shape, "CubeS")
        self.assertEqual(env.orientation, "normal")

    # OpenAI Gym Reset()
    def test_env_reset(self):
        """ Test the environment resets to the initial values (both from metadata and default initialization) """
        env_name = "gym_kinova_gripper:kinovagripper-v0"
        metadata = {"idx": 0, "Noise": True, "Orn": "normal", "Orn_Values": [0.01, 1.23, -3.4], "Shape": "CubeS",
                    "Start_Values": [1.1, -2.3, 0.0],
                    "xml_file": "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"}

        env = gym.make(env_name)  # Make initial environment

        env.reset(metadata=metadata)

    ## Test the hand location is correctly being determined
    def test_determine_hand_location(self):
        env_name = "gym_kinova_gripper:kinovagripper-v0"
        env = gym.make(env_name)  # Make initial environment

        # Test normal orientation
        metadata = {"idx": 0, "Noise": True, "Orn": "normal", "Orn_Values": [0.01, 1.23, -3.4], "Shape": "CubeS",
                    "Start_Values": [1.1, -2.3, 0.0],
                    "xml_file": "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"}
        env.reset(metadata=metadata)

        self.assertEqual(env.orientation, "normal")
        xloc, yloc, zloc, f1prox, f2prox, f3prox = env.determine_hand_location()
        self.assertEqual([xloc, yloc, zloc, f1prox, f2prox, f3prox], [0, 0, 0, 0, 0, 0])

        # Test top orientation
        # metadata["Orn"] = "top"
        # env.reset(metadata=metadata)
        # self.assertEqual(env.orientation, "top") # % TODO: Need clarification on the object size function

    def test_write_xml(self):
        """ Test the xml file contents are being re-written correctly (and the data stored in the mujco simulation is correct) """
        env_name = "gym_kinova_gripper:kinovagripper-v0"
        env = gym.make(env_name)  # Make initial environment

        # Test normal orientation
        metadata = {"idx": 0, "Noise": True, "Orn": "normal", "Orn_Values": [0.01, 1.23, -3.4], "Shape": "CubeS",
                    "Start_Values": [1.1, -2.3, 0.0],
                    "xml_file": "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"}
        env.reset(metadata=metadata)

        env.write_xml(new_rotation=[0,0,0])
        env.reset(metadata=metadata)
        # self.assertEqual( % TODO: Need help with navigating write_xml function to produce an accurate test case
        # self.wrist_pose=np.copy(self._sim.data.get_geom_xpos('palm'))

    def test_set_state(self):
        """ Test set_state() --> what we send Mujoco matches whats in MjData --> Getting it from the updated simulation """
        env_name = "gym_kinova_gripper:kinovagripper-v0"
        env = gym.make(env_name)  # Make initial environment

        # Test normal orientation
        metadata = {"idx": 0, "Noise": True, "Orn": "normal", "Orn_Values": [0.01, 1.23, -3.4], "Shape": "CubeS",
                    "Start_Values": [1.1, -2.3, 0.0],
                    "xml_file": "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"}
        env.reset(metadata=metadata)

        dummy_state = np.array([0, 0, 0, 0, 0, 0, 10, 10, 10])
        #self._model = load_model_from_path(self.file_dir + self.filename)
        #self._sim = MjSim(self._model)
        #env._set_state(dummy_state)
        #env.assertEqual(env._sim.data.qpos[0:9], dummy_state)

    # sim.Forward() (Make sure that before and after the forward step is what we expect)

    def test_env_step(self):
        """ Test the environment advances the simulation when performing a step() (OpenAI) """
        env_name = "gym_kinova_gripper:kinovagripper-v0"
        metadata = {"idx": 0, "Noise": True, "Orn": "normal", "Orn_Values": [0.01, 1.23, -3.4], "Shape": "CubeS",
                    "Start_Values": [1.1, -2.3, 0.0],
                    "xml_file": "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"}
        action = np.array([0, 0, 0, 0])

        env = gym.make(env_name)  # Make initial environment

        env.reset(metadata=metadata)

        for _ in range(10):
            _, _, done, _ = env.step(action)
            if done:
                env.reset(metadata=metadata)

    ## STEP() TESTING
        # Test the min-max of the action values sent to the simulator (this action should have a new name action_sim or something)
        # Move the setting of the action/sliders to a new function for testing
        # Have a test for the OpenAI step()
        # Have a test for the Mujoco Simulation step()

    ## MUJOCO SIMULATOR - ENVIRONMENT
        # Check the simulation has loaded (we should record the xml file used)
        # Check the simulation model is loaded (MjSim)
        # Check the viewer is initially set to None
        # Check the simulation model timestep is initially set to the default 0.01

        ''''
        self._model, self.obj_size, self.filename = load_model_from_path(self.file_dir + "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"), 's', "/kinova_description/j2s7s300_end_effector_v1_CubeS.xml"
        self._sim = MjSim(self._model)  # The simulator. This holds all the information about object locations and orientations
        self._viewer = None  # The render window
        self.contacts = self._sim.data.ncon  # The number of contacts in the simulation environment
        self._timestep = self._sim.model.opt.timestep
        self.site_count = 0 # Xml file
        self._simulator = "Mujoco"
        self.step_coords='global' # Might not be needed but CHECK! for simulation needs - referenced in Step() to determine coordinates
        '''

    ## OPEN AI GYM - ENVIRONMENT
        # Check the number of frames and episode steps are correct/match with what openai lists
        '''
        self.max_episode_steps = 30 # Maximum RL ime steps (Step() calls) within an episode
        self.frame_skip = frame_skip # Number of simulation frames run per RL time step
        self._numSteps = 0 # Count of Step() method calls
        self.arm_or_hand = "hand" # Remove after testing Step() function !!
        '''

    ## Other testing
        # Test the seed of the environment
        # Rendering

if __name__ == '__main__':
    unittest.main()

""" Nigel's testing code --> Sort through this
# Environment (simulation) testing
def test_self(self):
    shapes = ['Cube', 'Cylinder', 'Cone1', 'Cone2', 'Bowl', 'Rbowl', 'Bottle', 'TBottle', 'Hour', 'Vase', 'Lemon']
    sizes = ['S', 'M', 'B']
    keys = ["CubeS", "CubeB", "CylinderS", "CylinderB", "Cone1S", "Cone1B", "Cone2S", "Cone2B", "Vase1S", "Vase1B",
            "Vase2S", "Vase2B"]
    key = random.choice(keys)
    self.reset(obj_params=[key[0:-1], key[-1]])
    print('testing shape', key)
    self._get_obs()
    x = threading.Thread(target=self.obs_test)
    x.start()
    while x.is_alive():
        self.render()
    print('')
    print('testing step in global coords')
    action = [0, 0, 0, 0]
    self.step_coords = 'global'
    start_obs = self._get_obs(state_rep='global')
    for i in range(150):
        action[0] = np.random.rand() - 0.2
        self.step(action)
    end_obs = self._get_obs(state_rep='global')
    if (abs(start_obs[18] - end_obs[18]) > 0.001) | (abs(start_obs[19] - end_obs[19]) > 0.001):
        print('test failed. x/y position changed when it should not have, check step function')
    else:
        print('test passed')
    print('printing test step in local coords')
    self.reset(obj_params=[key[0:-1], key[-1]])
    self.step_coords = 'local'
    start_obs = self._get_obs()
    for i in range(150):
        action[0] = np.random.rand() - 0.2
        self.step(action)
    end_obs = self._get_obs()
    if (abs(start_obs[18] - end_obs[18]) > 0.001) | (abs(start_obs[19] - end_obs[19]) > 0.001):
        print('test failed. x/y position changed when it should not have, check step function')
    else:
        print('test passed')
    print('no current test for 6 axis motion, step tests finished.')
    print('begining shape test')
    bad_shapes = []
    for shape in shapes:
        for size in sizes:
            self.reset(obj_params=[shape, size])
            self.render()
            a = input('obj shape and size', shape, size, '. Is this correct y/n?')
            if a.lower() == 'y':
                print('shape passed')
            else:
                print('shape failed. recording')
                bad_shapes.append([shape, size])
    if bad_shapes == []:
        print('all shapes and sizes are accurate, tests finished')
    else:
        print('the following are shapes that were not correct. Look at the xml files.')
        print(bad_shapes)
"""
