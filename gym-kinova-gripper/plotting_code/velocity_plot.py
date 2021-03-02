import utils
import numpy as np
import glob
import os


# Print output stats
def print_stats(name, length, all_stats):
    """ Print the velocity stats collected (mins, maxs, means, etc.) """
    print("Evaluating: ", name)
    print("Length: ", length)

    for stat in all_stats:
        stat_text = ""
        for key, value in stat.items():
            stat_text += key + ": " + str(value) + " "
        print(stat_text)


# Analyze output of finger velocities (min, max, average)
def get_stats(f1_velocities, f2_velocities, f3_velocities):
    """ Get the min, max, range, and mean of finger velocity values
    f1_velocities: finger 1 velocities
    f2_velocities: finger 2 velocities
    f3_velocities: finger 3 velocities
    return: dictionaries containing min, max, range, and mean values per finger
    """
    # Init dictionaries to hold stats
    mins = {}
    maxs = {}
    ranges = {}
    means = {}

    # Minimum velocity
    mins["f1_min"] = min(f1_velocities)
    mins["f2_min"] = min(f2_velocities)
    mins["f3_min"] = min(f3_velocities)

    # Maximum velocity
    maxs["f1_max"] = max(f1_velocities)
    maxs["f2_max"] = max(f2_velocities)
    maxs["f3_max"] = max(f3_velocities)

    # Range of velocities
    ranges["f1_range"] = maxs["f1_max"] - mins["f1_min"]
    ranges["f2_range"] = maxs["f2_max"] - mins["f2_min"]
    ranges["f3_range"] = maxs["f3_max"] - mins["f3_min"]

    # Average finger velocity
    means["f1_mean"] = np.mean(f1_velocities)
    means["f2_mean"] = np.mean(f2_velocities)
    means["f3_mean"] = np.mean(f3_velocities)

    return mins, maxs, ranges, means


def evaluate_replay_velocities(name, replay_buffer):
    """ Evaluate mins, maxs, ranges, mean stats on policy action (finger velocity) output from replay buffer
        replay buffer: Replay buffer to evaluate
    """
    # Over all velocities within the buffer
    f1_velocities = np.array([])
    f2_velocities = np.array([])
    f3_velocities = np.array([])

    # Over single episode
    f1_ep_vels = np.array([])
    f2_ep_vels = np.array([])
    f3_ep_vels = np.array([])

    for episode_num in range(replay_buffer.replay_ep_num):
        episode_length = len(replay_buffer.reward[episode_num])
        for timestep_num in range(episode_length):
            # Get each finger velocity at the current time step
            f1_vel = replay_buffer.action[episode_num][timestep_num][1]
            f2_vel = replay_buffer.action[episode_num][timestep_num][2]
            f3_vel = replay_buffer.action[episode_num][timestep_num][3]

            # Append time step velocity to the list of episode finger velocities
            f1_ep_vels = np.append(f1_ep_vels, f1_vel)
            f2_ep_vels = np.append(f2_ep_vels, f2_vel)
            f3_ep_vels = np.append(f3_ep_vels, f3_vel)

        # Get the min, mean, max velocity values for the current episode
        ep_mins, ep_maxs, ep_ranges, ep_means = get_stats(f1_ep_vels, f2_ep_vels, f3_ep_vels)

        # Print out velocity data for the current episode
        # ep_stats = [ep_mins, ep_maxs, ep_ranges, ep_means]
        # print_stats("Episode "+str(episode_num), episode_length, ep_stats)

        # Append episode velocities to the full list of finger velocities over all episodes
        f1_velocities = np.append(f1_velocities, f1_ep_vels)
        f2_velocities = np.append(f2_velocities, f2_ep_vels)
        f3_velocities = np.append(f3_velocities, f3_ep_vels)

        # Post-episode analysis, clear array
        f1_ep_vels = np.array([])
        f2_ep_vels = np.array([])
        f3_ep_vels = np.array([])

    # All episodes/replay buffer content analysis
    all_mins, all_maxs, all_ranges, all_means = get_stats(f1_velocities, f2_velocities, f3_velocities)

    # Print out velocity data for over all episodes in buffer
    print("\n------ Analysis over ALL episodes ------\n")
    print("Replay Buffer: ", name)
    print("File path: ", replay_file_path)
    all_stats = [all_mins, all_maxs, all_ranges, all_means]
    print_stats("ALL episodes", replay_buffer.replay_ep_num, all_stats)


def get_replay_buffer_paths(full_path):
    """ Fill a replay buffer dictionary with all replay buffers within full_path """
    replay_paths = {}
    for path in glob.glob(full_path):
        name = os.path.basename(path)
        path = path.replace("\\", "/")
        #print("Name: ",name)
        #print("path: ",path)
        replay_paths[name] = path + "/"
    #print("replay_paths.items(): ",replay_paths.items())
    return replay_paths


if __name__ == "__main__":
    # Replay buffer initialization values for loading new buffer
    state_dim = 82
    action_dim = 4
    replay_size = 10000

    # Replay buffer file paths
    expert_no_grasp_path = "../expert_replay_data/Expert_data_NO_GRASP/"
    expert_w_grasp_path = "../expert_replay_data/Expert_data_WITH_GRASP/"
    batch_path = "../replay_buffer/2_18_STEPH_TEST_BATCH/replay_buffer_02_18_21_2100/"
    single_path = "../replay_buffer/2_18_STEPH_TEST_SINGLE_EP/replay_buffer_02_18_21_2258/"

    #all_replay_paths = {"Expert NO Grasp": expert_no_grasp_path, "Expert WITH Grasp": expert_w_grasp_path, "Batch": batch_path, "Single": single_path}

    """
    expert_pid_no_grasp_paths = get_replay_buffer_paths("../expert_replay_data_NO_NOISE/no_grasp/expert_naive/*")
    naive_pid_no_grasp_paths = get_replay_buffer_paths("../expert_replay_data_NO_NOISE/no_grasp/naive_only/*")

    expert_pid_with_grasp_paths = get_replay_buffer_paths("../expert_replay_data_NO_NOISE/with_grasp/expert_naive/*")
    naive_pid_with_grasp_paths = get_replay_buffer_paths("../expert_replay_data_NO_NOISE/with_grasp/naive_only/*")
    """

    test_naive_only_pid_no_grasp_path = get_replay_buffer_paths("../expert_replay_data_NO_NOISE/no_grasp/naive_only/*/normal/replay_buffer")
    test_expert_only_pid_no_grasp_path = get_replay_buffer_paths("../expert_replay_data_NO_NOISE/no_grasp/expert_only/*/normal/replay_buffer")
    test_expert_naive_pid_no_grasp_path = get_replay_buffer_paths("../expert_replay_data_NO_NOISE/no_grasp/expert_naive/*/normal/replay_buffer")

    #import copy
    #test_expert_naive_pid_no_grasp_dict = copy.deepcopy(test_expert_naive_pid_no_grasp_path)
    #print("test_expert_naive_pid_no_grasp_dict keys: ",test_expert_naive_pid_no_grasp_dict.keys())
    #print("test_expert_naive_pid_no_grasp_dict values: ", test_expert_naive_pid_no_grasp_dict.values())

    all_replay_paths = {"Test Naive Only": test_naive_only_pid_no_grasp_path, "Test Expert Only": test_expert_only_pid_no_grasp_path, "Test Expert Niave: ": test_expert_naive_pid_no_grasp_path}

    # Evaluate each replay buffer for its reward makeup and policy output action (finger velocity) stats
    for replay_path_type, replay_path_dict in all_replay_paths.items():
        print("\n\n========== Evaluating ", replay_path_type," ===========")
        print("Evaluating replay_path_dict", replay_path_dict)
        for name, replay_file_path in replay_path_dict.items():
            print("name: ", name, "\nreplay_file_path: ", replay_file_path)
            # Load the replay buffer (expert replay buffer)
            replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, replay_size)
            replay_text = replay_buffer.store_saved_data_into_replay(replay_file_path)

            # Evaluate policy action output (finger velocities) from replay buffer
            evaluate_replay_velocities(name, replay_buffer)