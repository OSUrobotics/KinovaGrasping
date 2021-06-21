import utils    # Import Replay Buffer (located in utils)
import matplotlib.pyplot as plt     # Plotting library
import matplotlib.patches as patches    # Allows for text boxes to be added to the plot
import numpy as np  # Numpy arrays
import argparse # Command-line parsing
import copy # Used to copy replay buffer contents to an array for plotting

def get_stats(metric_arr):
    """ Get the min, max, range, and average over the input array """
    min_value = min(metric_arr)
    max_value = max(metric_arr)
    range_value = max_value - min_value
    mean_value = np.mean(metric_arr)

    return min_value, max_value, range_value, mean_value

def actual_values_plot(metrics_arr, episode_idx, label_name, metric_name, saving_dir=None):
    """ Create a simple plot with the actual metric values """
    fig = plt.figure()
    plt.plot(metrics_arr[episode_idx],label=label_name)
    plt.title("Actual Values of "+ str(metric_name) + " over episode" + str(episode_idx+1))
    plt.xlabel("Time step")
    plt.ylabel(str(metric_name))
    plt.legend(loc='best')
    plt.ylim([0, 0.8])

    if saving_dir is not None:
        plt.savefig(saving_dir + "/" + label_name + "act_vals" + ".png")
        plt.close(fig)

def episode_distribution_plot(metrics_arr):
    """ Plot distribution over one episode """
    plt.hist(metrics_arr)
    plt.title("Distribution over an episode")
    plt.show()


def all_episodes_average_plot(range_of_episodes, metrics_arr, label_name, metric_name, axes_limits, saving_dir=None):
    """ Creates an average plot displaying metric values over a range of episodes
    range_of_episodes: Numpy array containing desired episode indexes
    metrics_arr: Numpy array containing metrics to be plotted (state/action metrics, reward metrics) for each episode
    metric_name: Name (str) of the plotted metric
    """
    actual_values = [] # Used to calculate min/max over all output
    averages = []
    # Calculate the average metric value over an episode
    for idx in range(len(range_of_episodes)):
        actual_values.extend(metrics_arr[idx])
        avg = np.mean(metrics_arr[idx])
        averages.append(avg)

    # Plot average values
    plt.plot(range_of_episodes, averages, label=label_name)
    plt.title("Average value of "+ metric_name + " per episode over episodes" + str(min(range_of_episodes)+1) + "-" + str(max(range_of_episodes)+1))
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.legend(loc='best')
    plt.xlim([axes_limits['x_min'], axes_limits['x_max']])
    plt.ylim([axes_limits['y_min'], axes_limits['y_max']])

    # Get general stats about metric output
    min_value, max_value, range_value, mean_value = get_stats(actual_values)

    # Return text string to be placed on plot (with stats about the metric)
    textstr = "\n{}:\n".format(label_name)
    textstr += "Mean: {:.3f}\nMin: {:.3f}\nMax: {:.3f}\nRange: {:.3f}\n".format(mean_value,min_value,max_value,range_value)

    if saving_dir is not None:
        plt.savefig(saving_dir + "/" + metric_name + "average" + ".png")

    return textstr


def all_episodes_distribution_plot(range_of_episodes, metrics_arr, metric_name, saving_dir=None):
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
    if saving_dir is not None:
        plt.savefig(saving_dir + "/" + metric_name + "distribution" + ".png")


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


def get_selected_episode_metrics(range_of_episodes,metric_arr,metric_idx):
    """ Step through the replay buffer for each episode and evaluate through stats or plotting
        range_of_episodes: List of episode indexes within our desired episode range
        replay_buffer: Replay buffer containing (s,a,ns,r,d) transitions
        Each state/action/next_state/reward/done value is a list of episode values; for example:
            replay_buffer.action is in the following format: [ [np.array([]),...,np.array([])], [np.array([]),...,np.array([])] ]
            where replay_buffer.action is a list of episodes
            where each episode is a list of numpy arrays containing the action values per transition
        Return selected_episodes -- A full list of metric values from within the selected range of episodes
    """
    selected_episodes = []

    for episode_idx in range_of_episodes:
        # Get all metrics from an episode
        episode_metric = metric_arr[episode_idx]

        # Convert each episode (list of transitions) into a stacked numpy array for manipulation
        episode_metric = np.stack(episode_metric, axis=0)

        # Append the desired metric (at the metric_idx) from the current episode to a full list
        selected_episodes.append(episode_metric[:, metric_idx].tolist())

    return selected_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_filepaths", type=str, action='store', default=None) # List of the replay buffer filepaths to compare pver
    parser.add_argument("--metric_filepaths", type=str, action='store', default=None) # List of the metric array filepaths to compare over
    parser.add_argument("--label_names", type=str, action='store', default="No Label") # List of labels for each plotted line
    parser.add_argument("--metric_name", type=str, action='store', default=None) # Name of the metric being evaluated (ex: Finger 1 Velocity)
    parser.add_argument("--metric_idx", type=int, action='store', default=None) # Index of the metric being evaluated (Ex: idx = 1, ReplayBuffer.action[idx] == Finger 1 Velocity)
    parser.add_argument("--episode_idx", type=int, action='store', default=0) # Index of a specific episode to evaluate
    parser.add_argument("--plot_type", type=str, action='store', default="average") # Type of plot to be produced; Options: average, actual, distribution, boxplot
    parser.add_argument("--saving_dir", type=str, action='store', default=None) # Directory where plot output is saved

    args = parser.parse_args()

    state_dim = 82
    action_dim = 4
    replay_size = 4000 # Size of the replay buffer (# Episodes)
    replay_filepaths = args.replay_filepaths
    metric_filepaths = args.metric_filepaths
    label_names = args.label_names.split(',')
    metric_idx = args.metric_idx
    episode_idx = args.episode_idx
    metric_name = args.metric_name
    plot_type = args.plot_type
    saving_dir = args.saving_dir

    ## For reference:
    #Constant-controller replay buffer: "../replay_buffer/expert_replay_data_NO_NOISE/no_grasp/naive_only/CubeS/normal/replay_buffer/"
    #Pre-trained policy replay buffer: "../replay_buffer/BC_4keps/replay_buffer_04_15/"

    metric_arrays = []

    if replay_filepaths is not None:
        replay_filepaths = replay_filepaths.split(',')  # Sets list of replay buffer filepaths
        for filepath in replay_filepaths:
            replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, replay_size) # Load the replay buffer (expert replay buffer)
            replay_text = replay_buffer.store_saved_data_into_replay(filepath) # Load stored data into the initialized replay buffer
            replay_buffer_array = copy.deepcopy(replay_buffer.action)
            metric_arrays.append(replay_buffer_array)

    if metric_filepaths is not None:
        metric_filepaths = metric_filepaths.split(',')  # Sets list of metric filepaths
        for filepath in metric_filepaths:
            metric_arrays.append(np.load(filepath, allow_pickle=True))

    plt.figure() # Create the figure before plotting to allow multiple lines to be plotted
    textstr = "" # Text to be displayed on the plot (Ex: Min, Max, Avg. info per metric)
    axes_limits = {"x_min": 0, "x_max": replay_size, "y_min": 0, "y_max": 0.9}

    for label_name, metric_arr in zip(label_names,metric_arrays):
        start_episode_idx = 0
        end_episode_idx = len(metric_arr)
        range_of_episodes = np.arange(start_episode_idx, end_episode_idx) # List of selected range of episode indexes

        # Get selected episode metrics from the data over all episodes
        selected_metric_arr = get_selected_episode_metrics(range_of_episodes,metric_arr=metric_arr, metric_idx=metric_idx)

        min_arr_value = min([min(metric_arr) for metric_arr in selected_metric_arr])
        max_arr_value = max([max(metric_arr) for metric_arr in selected_metric_arr])
        axes_limits = {"x_min": start_episode_idx, "x_max": end_episode_idx, "y_min": min_arr_value, "y_max": max_arr_value}

        # Plot the metrics based on the desired plot type
        if plot_type == "average":
            textstr += all_episodes_average_plot(range_of_episodes, selected_metric_arr, label_name, metric_name, axes_limits)
        elif plot_type == "actual":
            actual_values_plot(selected_metric_arr, episode_idx, label_name, metric_name)
        elif plot_type == "distribution":
            all_episodes_distribution_plot(range_of_episodes, selected_metric_arr, metric_name)
        elif plot_type == "boxplot":
            all_episodes_boxplot(range_of_episodes, selected_metric_arr, metric_name, freq=200, min_val=0, max_val=0.9)

    # Set matplotlib.patch.Patch properties
    props = dict(facecolor='white', pad=5, alpha=0.5)
    # Place a text box in the upper left corner
    plt.figtext(axes_limits["x_min"]+0.01, axes_limits["y_max"], textstr, fontsize=7,
            horizontalalignment='left', verticalalignment='top', bbox=props)
    plt.subplots_adjust(left=0.2)

    if saving_dir is None:
        plt.show()
    else:
        plt.savefig(saving_dir+"/"+metric_name+plot_type+".png")