## Isolate and test certain functionality within RL algorithm ##

import argparse

import numpy as np
import torch
import gym

import utils
import DDPGfD

import time
import stable_baselines3

from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise



def test_naive_controller(test_args=None):
    """ Test the naive controller input and output
    test_args: Specific arguments to be altered from default for testing
    """
    print("Testing the Naive Controller")
    # Pass in arguments we would like to test, otherwise arguments are set to default
    # FROM MAIN_DDPG setup_args(args)
    # Call mode naive from main_DDPG
    # Run for set number of episodes
    # Examine output from generated plots
    # Add extra test cases, checks, output analysis where needed


def test_training(test_args=None):
    """ Test the RL policy training input and output
    test_args: Specific arguments to be altered from default for testing
    """
    print("Testing Policy training")
    # Pass in arguments we would like to test, otherwise arguments are set to default
    # FROM MAIN_DDPG setup_args(args)
    # Call mode train from main_DDPG
    # Run for set number of episodes
    # Examine output from generated plots
    # Add extra test cases, checks, output analysis where needed

#def get_test_args(filename) # Get arguments to test with from file
#def test_replay_buffer():   # Test replay buffer
#def test_training_sample(): # Test policy sampling
#def test_training_update(): # Test policy update


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, action='store', default=10)
    parser.add_argument("--test_mode", type=str, action='store', default="train")
    parser.add_argument("--shapes", default='CubeS', action='store', type=str)  # Requested shapes to use
    parser.add_argument("--orientation", type=str, action='store', default="normal")
    parser.add_argument("--policy_name", default="DDPGfD")              # Policy name
    parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0") # OpenAI gym environment name
    args = parser.parse_args()

    print("------ Testing ", args.test_mode," ------")

    env_name = 'gym_kinova_gripper:kinovagripper-v0'
    # env_name = 'InvertedPendulum-v2'

    env = gym.make(env_name)

    from stable_baselines3.common.env_checker import check_env
    check_env(env)

    shape_keys = 'CubeS'
    hand_orientation = 'store'

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    print("========== env info")
    print(state_dim)
    print(action_dim)
    print(max_action)

    agent_replay_size = 1000

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "n": 5,
        "discount": 0.99,
        "tau": 0.0005,
        "batch_size": 1
    }

    # policy = DDPGfD.DDPGfD(**kwargs)
    #
    # replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, agent_replay_size)


    # state = env.reset(shape_keys=shape_keys, hand_orientation=hand_orientation)
    # with_grasp: whether you get a grasp reward or not...
    # state = env.reset(shape_keys=['CubeS'], with_grasp=False,hand_orientation='normal',mode='train',env_name="eval_env")
    state = env.reset()

    print("===========random info about reset state")
    print(type(state))
    # print(state)
    print(state.shape)
    print("============================")

    # done = False
    #
    # while not done:
    #     print("taking a step...")
    #     action = env.action_space.sample()
    #
    #     next_state, reward, done, _ = env.step(action)
    #
    #     print("AAAAAAA")
    #     print(type(next_state))
    #     print(next_state)
    #
    #     # to visualize.
    #     env.render()
    #     time.sleep(0.05)
    #
    #     state = next_state


    # action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))
    #
    # model = DDPG('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=10000, log_interval=10)
    # model.save("ddpg_test1")
    #
    # env = model.get_env()
    #
    # del model  # remove to demonstrate saving and loading

    model = DDPG.load("ddpg_test1")

    while True:
        obs = env.reset()
        print("hi")
        done = False
        while not done:
            action, _states = model.predict(obs)
            print(action)
            next_obs, rewards, done, info = env.step(action)

            # render env
            env.render()
            time.sleep(0.05)

            obs = next_obs

    print("finished testing")




