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

def episode_distribution_plot(metrics_arr):
    """ Plot distribution over one episode """
    plt.hist(metrics_arr)
    plt.title("Distribution over an episode")
    plt.show()


def all_episodes_average_plot(range_of_episodes, metrics_arr, metric_name):
    """ Creates an average plot displaying metric values over a range of episodes
    range_of_episodes: Numpy array containing desired episode indexes
    metrics_arr: Numpy array containing metrics to be plotted (state/action metrics, reward metrics) for each episode
    metric_name: Name (str) of the plotted metric
    """
    averages = []
    # Calculate the average metric value over an episode
    for idx in range(len(range_of_episodes)):
        avg = np.mean(metrics_arr[idx])
        averages.append(avg)

    # Plot average values
    plt.plot(range_of_episodes, averages)
    plt.title("Average value of "+ metric_name + " over episodes " + str(min(range_of_episodes)) + "-" + str(max(range_of_episodes)))
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.show()


def all_episodes_distribution_plot(range_of_episodes, metrics_arr, metric_name):
    """ Creates a distribution plot displaying metric values over a range of episodes
    range_of_episodes: Numpy array containing desired episode indexes
    metrics_arr: Numpy array containing metrics to be plotted (state/action metrics, reward metrics) for each episode
    metric_name: Name (str) of the plotted metric
    """
    # Plot the distribution of metric values for each episode
    idx = 0
    for episode_num in range_of_episodes:
        plt.hist(metrics_arr[idx], label=str(episode_num))
        idx += 1

    # Create plot labels
    plt.legend(title="Episode", loc='upper right')
    plt.title("Distribution of "+ metric_name + " over episodes " + str(min(range_of_episodes)) + "-" + str(max(range_of_episodes)))
    plt.xlabel(metric_name)
    plt.ylabel("Occurrences within the episode")
    plt.ylim(top=30) # Max number of times steps in an episode

    # Show plot
    plt.show()


def all_episodes_boxplot(range_of_episodes, metrics_arr, metric_name, freq, min_val, max_val):
    """ Creates a boxplot displaying metric values over a range of episodes
    range_of_episodes: Numpy array containing desired episode indexes
    metrics_arr: Numpy array containing metrics to be plotted (state/action metrics, reward metrics) for each episode
    metric_name: Name (str) of the plotted metric
    freq: Frequency of boxplots (over the desired episode range)
    """
    # Get all episode indexes per frequency
    indexes = np.arange(len(range_of_episodes))
    selected_indexes = indexes[::freq]
    selected_episode_nums = range_of_episodes[::freq]

    # Create boxplots
    bp = plt.boxplot(np.asarray(metrics_arr)[selected_indexes])

    # Create accurate tick labeling and plot labels
    x_tick_locs = np.linspace(1, len(selected_episode_nums), len(selected_episode_nums))
    plt.xticks(ticks=x_tick_locs, labels=selected_episode_nums)
    plt.title(metric_name + " per " + str(freq) + " episodes within episodes " + str(min(range_of_episodes)) + "-" + str(max(range_of_episodes)+ 10001))
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.ylim(bottom=min_val,top=max_val)

    # Show plot in new window
    plt.show()


def evaluate_replay_buffer(replay_buffer):
    """ Step through the replay buffer for each episode and evaluate through stats or plotting
        replay_buffer: Replay buffer containing (s,a,ns,r,d) transitions
        Each state/action/next_state/reward/done value is a list of episode values; for example:
            replay_buffer.action is in the following format: [ [np.array([]),...,np.array([])], [np.array([]),...,np.array([])] ]
            where replay_buffer.action is a list of episodes
            where each episode is a list of numpy arrays containing the action values per transition
    """
    start_episode_idx = 0
    end_episode_idx = 1998
    range_of_episodes = np.arange(start_episode_idx, end_episode_idx + 1) # +1 to include the ending index
    #sample_size = 100
    #random_sample = np.random.choice(range_of_episodes, sample_size)

    selected_episodes = []

    # EVALUATE OVER RANGE OF EPISODES
    for episode_idx in range_of_episodes: # FROM THE RANGE OF EPISODES WE'RE INTERESTED IN

        episode_action = replay_buffer.action[episode_idx] # GET ALL ACTIONS FROM THAT EPISODE

        # Convert each episode (list of transitions) into a stacked numpy array for manipulation
        episode_action = np.stack(episode_action, axis=0)

        # Plot values over from episode
        #actual_values_plot(episode_action[:, 1])
        #episode_distribution_plot(episode_action[:, 1])

        # Finger 1 velocities
        selected_episodes.append(episode_action[:, 3]) # APPEND ALL ACTIONS FROM ONE EPISODE, APPEND TO FULL LIST

    # PLOT OVER ALL EPISODES IN SELECTED RANGE (AVG, DISTRIBUTION, BOXPLOT)
    # Plotting finger 1 velocities over all episodes
    metric_name = "Finger 3 Velocity"
    all_episodes_average_plot(range_of_episodes, selected_episodes, metric_name)
    #all_episodes_distribution_plot(range_of_episodes, selected_episodes, metric_name)
    all_episodes_boxplot(range_of_episodes, selected_episodes, metric_name, freq=200, min_val=0, max_val=0.9)

if __name__ == "__main__":
    # Replay buffer initialization values for loading new buffer
    state_dim = 82
    action_dim = 4
    replay_size = 20000

    # Naive replay buffer
    replay_filepath = "../replay_buffer/expert_replay_data_NO_NOISE/no_grasp/naive_only/CubeS/normal/replay_buffer/"
    # "../replay_buffer/BC_4keps/replay_buffer_04_15/"

    # Load the replay buffer (expert replay buffer)
    replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, replay_size)

    # Load stored data into the initialized replay buffer
    replay_text = replay_buffer.store_saved_data_into_replay(replay_filepath)

    # Go through replay buffer
    evaluate_replay_buffer(replay_buffer)