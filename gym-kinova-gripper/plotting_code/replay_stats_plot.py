import utils    # Import Replay Buffer (located in utils)
import matplotlib.pyplot as plt     # Plotting library
import numpy as np  # Numpy arrays

def get_stats():
    """ Get the min, max, range, and average over the input array """

def actual_values_plot(metrics_arr):
    """ Create a simple plot with the actual metric values """
    plt.plot(metrics_arr)
    plt.title("Actual Values")
    plt.show()

def average_plot(metrics_arr):
    """ Plot average """
    avg = np.mean(metrics_arr)
    print("avg: ",avg)
    plt.plot(avg)
    plt.title("Average")
    plt.show()

def episode_distribution_plot(metrics_arr):
    """ Plot distribution over one episode """
    plt.hist(metrics_arr)
    plt.title("Distribution")
    plt.show()

def all_episodes_distribution_plot():
    """ Plot distribution over all episodes """

def evaluate_replay_buffer(replay_buffer):
    """ Step through the replay buffer for each episode and evaluate through stats or plotting
        replay_buffer: Replay buffer containing (s,a,ns,r,d) transitions
    """
    start_episode_idx = 0
    end_episode_idx = 5
    range_of_episodes = np.arange(start_episode_idx, end_episode_idx)

    # Replay_buffer.action is a list of episodes (list of transitions) --> list of lists
    for episode_idx in range_of_episodes:
        # Get action from the desired episode
        episode_action = replay_buffer.action[episode_idx]

        # Convert each episode (list of transitions) into a stacked numpy array for manipulation
        episode_action = np.stack(episode_action, axis=0)
        print("POST VSTACK episode_action.shape: ", episode_action.shape)

        # Plot values over from episode
        print("Plotting finger 1 velocities")
        actual_values_plot(episode_action[:, 1])
        average_plot(episode_action[:, 1])
        episode_distribution_plot(episode_action[:, 1])


if __name__ == "__main__":
    # Replay buffer initialization values for loading new buffer
    state_dim = 82
    action_dim = 4
    replay_size = 10000

    # Naive replay buffer
    replay_filepath = "../expert_replay_data_NO_NOISE/no_grasp/naive_only/CubeS/normal/replay_buffer/"

    # Load the replay buffer (expert replay buffer)
    replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, replay_size)

    # Load stored data into the initialized replay buffer
    replay_text = replay_buffer.store_saved_data_into_replay(replay_filepath)

    # Go through replay buffer
    evaluate_replay_buffer(replay_buffer)