## Isolate and test certain functionality within RL algorithm ##

import argparse

import numpy as np
import torch
import gym

import utils
import DDPGfD

import time
import stable_baselines3

from stable_baselines3 import DDPG, SAC, TD3, HER
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def check_oob(obs):
    # Originally used for defining min/max ranges of state input (currently not being used)
    # Originally used for defining min/max ranges of state input (currently not being used)
    min_hand_xyz = [-0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1,
                    -0.1, -0.1, -0.1, -0.1]
    min_obj_xyz = [-0.1, -0.01, -0.01]
    min_joint_states = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
    min_obj_size = [0.0, 0.0, 0.0]
    min_finger_obj_dist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    min_obj_dot_prod = [0.0]
    min_f_dot_prod = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    min_x_y = [-np.pi, -np.pi]
    min_rangefinder_data = 17 * [0.0]
    min_gravity_vector_local = 3 * [-1.0]
    min_obj_location = [-0.1, 0.0, -0.1]
    min_ratio_area_side = [0.0]
    min_ratio_area_top = [0.0]
    min_finger_dot_prod = 6 * [-1.0]
    min_wrist_dot_prod = [-1.0]

    max_hand_xyz = [0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1,
                    0.5]
    max_obj_xyz = [0.1, 0.7, 0.5]
    max_joint_states = [0.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    max_obj_size = [0.5, 0.5, 0.5]
    max_finger_obj_dist = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    max_obj_dot_prod = [1.0]
    max_f_dot_prod = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    max_x_y = [np.pi, np.pi]
    max_rangefinder_data = 17 * [6.0]
    max_gravity_vector_local = 3 * [1.0]
    max_obj_location = [0.2, 0.2, 0.2]
    max_ratio_area_side = [1.0]
    max_ratio_area_top = [1.0]
    max_finger_dot_prod = 6 * [1.0]
    max_wrist_dot_prod = [1.0]

    obs_min = min_hand_xyz + min_obj_xyz + min_joint_states + min_obj_size + min_finger_obj_dist + min_obj_dot_prod + \
              min_x_y + min_rangefinder_data + min_gravity_vector_local + min_obj_location + min_ratio_area_side + \
              min_ratio_area_top + min_finger_dot_prod + min_wrist_dot_prod

    obs_min = np.array(obs_min)

    obs_max = max_hand_xyz + max_obj_xyz + max_joint_states + max_obj_size + max_finger_obj_dist \
              + max_obj_dot_prod + max_x_y + max_rangefinder_data + max_gravity_vector_local + \
              max_obj_location + max_ratio_area_side + max_ratio_area_top + max_finger_dot_prod \
              + max_wrist_dot_prod
    obs_max = np.array(obs_max)

    print(obs.shape)

    return np.less_equal(obs_min, obs), np.less_equal(obs, obs_max)


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

    # this checks the environment for proper bounds...
    # from stable_baselines3.common.env_checker import check_env
    # check_env(env)

    shape_keys = 'CubeS'
    hand_orientation = 'store'

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    """
    copied directly from kinova_gripper_env.py
    """

    # Originally used for defining min/max ranges of state input (currently not being used)
    # Originally used for defining min/max ranges of state input (currently not being used)
    min_hand_xyz = [-0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1,
                    -0.1, -0.1, -0.1, -0.1]
    min_obj_xyz = [-0.1, -0.01, -0.01]
    min_joint_states = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
    min_obj_size = [0.0, 0.0, 0.0]
    min_finger_obj_dist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    min_obj_dot_prod = [0.0]
    min_f_dot_prod = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    min_x_y = [-np.pi, -np.pi]
    min_rangefinder_data = 17 * [0.0]
    min_gravity_vector_local = 3 * [-1.0]
    min_obj_location = [-0.1, 0.0, -0.1]
    min_ratio_area_side = [0.0]
    min_ratio_area_top = [0.0]
    min_finger_dot_prod = 6 * [-1.0]
    min_wrist_dot_prod = [-1.0]

    max_hand_xyz = [0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1,
                    0.5]
    max_obj_xyz = [0.1, 0.7, 0.5]
    max_joint_states = [0.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    max_obj_size = [0.5, 0.5, 0.5]
    max_finger_obj_dist = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    max_obj_dot_prod = [1.0]
    max_f_dot_prod = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    max_x_y = [np.pi, np.pi]
    max_rangefinder_data = 17 * [6.0]
    max_gravity_vector_local = 3 * [1.0]
    max_obj_location = [0.2, 0.2, 0.2]
    max_ratio_area_side = [1.0]
    max_ratio_area_top = [1.0]
    max_finger_dot_prod = 6 * [1.0]
    max_wrist_dot_prod = [1.0]

    obs_min = min_hand_xyz + min_obj_xyz + min_joint_states + min_obj_size + min_finger_obj_dist + min_obj_dot_prod + \
              min_x_y + min_rangefinder_data + min_gravity_vector_local + min_obj_location + min_ratio_area_side + \
              min_ratio_area_top + min_finger_dot_prod + min_wrist_dot_prod

    obs_min = np.array(obs_min)

    obs_max = max_hand_xyz + max_obj_xyz + max_joint_states + max_obj_size + max_finger_obj_dist \
              + max_obj_dot_prod + max_x_y + max_rangefinder_data + max_gravity_vector_local + \
              max_obj_location + max_ratio_area_side + max_ratio_area_top + max_finger_dot_prod \
              + max_wrist_dot_prod
    obs_max = np.array(obs_max)


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

    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    state = next_state

    print("===========random info about reset state")
    print(type(state))
    print(state.shape)
    print("============================")

    oob_arr_less, oob_arr_more = check_oob(state)
    print(oob_arr_less)
    print("============= more array")
    print(oob_arr_more)
    print('indices')
    low_violation = np.where(oob_arr_less == False)
    high_violation = np.where(oob_arr_more == False)
    print(low_violation)
    print(high_violation)

    print("low violating values (bounds and state)")
    print(obs_min[low_violation])
    print(state[low_violation])
    print("high violating values (bounds and state)")
    print(obs_max[high_violation])
    print(state[high_violation])


    print(1/0)

    done = False

    while not done:
        print("taking a step...")
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)

        print("AAAAAAA")
        print(type(next_state))
        print(next_state)

        # to visualize.
        env.render()
        time.sleep(0.05)

        state = next_state


    action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))

    # model = DDPG('MlpPolicy', env, verbose=1)
    model = TD3('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000000, log_interval=10)
    model.save("td3_test1")

    env = model.get_env()

    del model  # remove to demonstrate saving and loading

    model = DDPG.load("td3_test1")

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




