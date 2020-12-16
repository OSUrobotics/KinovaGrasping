import numpy as np
import random
import math
import os
import datetime
from expert_data import store_saved_data_into_replay
import glob
import matplotlib.pyplot as plt
import utils

def split_replay(replay_buffer=None,save_dir=None):
    print("In Split replay")

    if replay_buffer is None:
        replay_buffer = utils.ReplayBuffer_Queue(82, 4, 10100)
        # Default expert pid file path
        expert_file_path = "./expert_replay_data/Expert_data_11_18_20_0253/"
        print("Expert PID filepath: ", expert_file_path)
        replay_buffer = store_saved_data_into_replay(replay_buffer, expert_file_path)

    # Check and create directory
    expert_replay_saving_dir = "./expert_replay_data_split"
    if not os.path.isdir(expert_replay_saving_dir):
        os.mkdir(expert_replay_saving_dir)

    # Size of replay buffer data subset
    subset_size = 100

    # If number of episodes does not split evenly into 100, round up
    num_subsets = math.ceil(replay_buffer.replay_ep_num / subset_size)

    print("in SPLIT REPLAY, replay_buffer.replay_ep_num: ",replay_buffer.replay_ep_num)
    print("in SPLIT REPLAY, num_subsets: ",num_subsets)

    for idx in range(int(num_subsets)):
        print("**** idx: ",idx)
        print("idx + subset_size: ",idx + subset_size)
        # Get the final subset of episodes (even if it is a bit smaller than the subset size)
        if replay_buffer.replay_ep_num < idx + subset_size:
            subset_size = replay_buffer.replay_ep_num - idx

        # Get the beginning timestep index and the ending timestep index within an episode
        selected_indexes = np.arange(0, idx*subset_size)

        # Get subset batch of replay buffer data
        state_subset = [replay_buffer.state[x] for x in selected_indexes]
        action_subset = [replay_buffer.action[x] for x in selected_indexes]
        next_state_subset = [replay_buffer.next_state[x] for x in selected_indexes]
        reward_subset = [replay_buffer.reward[x] for x in selected_indexes]
        not_done_subset = [replay_buffer.not_done[x] for x in selected_indexes]

        # Set filename for subset
        state_filename = "state_" + str(idx) + ".npy"
        action_filename = "action_" + str(idx) + ".npy"
        next_state_filename = "next_state_" + str(idx) + ".npy"
        reward_filename = "reward_" + str(idx) + ".npy"
        not_done_filename = "node_done_" + str(idx) + ".npy"

        curr_save_dir = "Expert_data_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M")

        if not os.path.exists(os.path.join(expert_replay_saving_dir, curr_save_dir)):
            os.makedirs(os.path.join(expert_replay_saving_dir, curr_save_dir))

        save_filepath = expert_replay_saving_dir + "/" + curr_save_dir + "/"
        print("save_filepath: ", save_filepath)

        np.save(save_filepath + state_filename, state_subset)
        np.save(save_filepath + action_filename, action_subset)
        np.save(save_filepath + next_state_filename, next_state_subset)
        np.save(save_filepath + reward_filename, reward_subset)
        np.save(save_filepath + not_done_filename, not_done_subset)

        np.save(save_filepath + "episodes", replay_buffer.episodes) # Keep track of episode start/finish indexes
        np.save(save_filepath + "episodes_info", [replay_buffer.max_episode, replay_buffer.size, replay_buffer.episodes_count, replay_buffer.replay_ep_num])

        return save_filepath

def load_split_replay(replay_buffer=None, filepath=None):
    print("#### Getting SPLIT expert replay buffer from SAVED location: ",filepath)
    expert_state = []
    expert_action = []
    expert_next_state = []
    expert_reward = []
    expert_not_done = []

    for state_subset in glob.glob(filepath + '/state_*.npy'):
        state_subset = filepath + '/' + os.path.basename(state_subset)
        print("getting state file: ", state_subset)
        state = np.load(state_subset, allow_pickle=True).tolist()
        print("state: ", state)
        expert_state.append(state)
        print("expert_state: ",expert_state)
    for action_subset in glob.glob(filepath + '/action_*.npy'):
        expert_action.append(np.load(action_subset, allow_pickle=True).tolist())
    for next_state_subset in glob.glob(filepath + '/next_state_*.npy'):
        expert_next_state.append(np.load(next_state_subset, allow_pickle=True).tolist())
    for reward_subset in glob.glob(filepath + '/reward_*.npy'):
        expert_reward.append(np.load(reward_subset, allow_pickle=True).tolist())
    for not_done_subset in glob.glob(filepath + '/not_done_*.npy'):
        expert_not_done.append(np.load(not_done_subset, allow_pickle=True).tolist())

    expert_episodes = np.load(filepath + "/episodes.npy", allow_pickle=True).tolist()  # Keep track of episode start/finish indexes
    expert_episodes_info = np.load(filepath + "/episodes_info.npy", allow_pickle=True)

    replay_buffer.state = expert_state
    replay_buffer.action = expert_action
    replay_buffer.next_state = expert_next_state
    replay_buffer.reward = expert_reward
    replay_buffer.not_done = expert_not_done
    replay_buffer.episodes = expert_episodes

    replay_buffer.max_episode = expert_episodes_info[0]
    replay_buffer.size = expert_episodes_info[1]
    replay_buffer.episodes_count = expert_episodes_info[2]
    replay_buffer.replay_ep_num = expert_episodes_info[3]

    # max_episode: Maximum number of episodes, limit to when we remove old episodes
    # size: Full size of the replay buffer (number of entries over all episodes)
    # episodes_count: Number of episodes that have occurred (may be more than max replay buffer side)
    # replay_ep_num: Number of episodes currently in the replay buffer

    num_episodes = len(expert_state)
    print("num_episodes: ", num_episodes)

    return replay_buffer

def plot_reward_distribution(replay_buffer,saved_filepath):
    print("In plot reward distribution")
    n_bins = 20
    reward = np.load("./expert_replay_data/Expert_data_11_18_20_0253/reward.npy", allow_pickle=True).transpose()
    print("Loaded the reward")

    #plt.hist(replay_buffer.reward, bins=n_bins)
    plt.hist(reward.tolist(), bins=n_bins)
    plt.title("Reward distribution within replay buffer", weight='bold')
    plt.ylabel('# of reward values within expert replay')
    plt.xlabel('Reward value')
    plt.xlim(-1, 1)
    print("Plotted the figure")
    saved_filepath = "./expert_replay_data/Expert_data_11_18_20_0253/"
    plt.savefig(saved_filepath + "/new_reward_distribution")
    plt.clf()

#saved_filepath = split_replay(replay_buffer=None,save_dir=None)
#print("saved filepath: ",saved_filepath)
#print("After split_replay")
#replay_buffer_from_split = load_split_replay(replay_buffer=None,filepath=saved_filepath)
#print("After load_split_replay")

# Testing replay buffer output
#replay_buffer = utils.ReplayBuffer_Queue(82, 4, 10100)
# Default expert pid file path
#expert_file_path = "./expert_replay_data/Expert_data_11_18_20_0253/"
#print("Expert PID filepath: ", expert_file_path)
#replay_buffer = store_saved_data_into_replay(replay_buffer, expert_file_path)
#plot_reward_distribution(replay_buffer=None,saved_filepath=None)

