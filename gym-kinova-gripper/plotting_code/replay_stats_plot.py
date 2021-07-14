import utils    # Import Replay Buffer (located in utils)
import matplotlib.pyplot as plt     # Plotting library
import matplotlib.patches as patches    # Allows for text boxes to be added to the plot
import numpy as np  # Numpy arrays
import argparse # Command-line parsing
import copy # Used to copy replay buffer contents to an array for plotting
import pandas as pd
from matplotlib.ticker import MultipleLocator

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

def actual_values_plot_all_episodes(range_of_episodes, metrics_arr, metric_name, saving_dir=None):
    """ Create a simple plot with the actual metric values """
    actual_values = [] # Used to calculate min/max over all output
    timestep_labels = [] # x-axis labels
    timestep_major_locs = [] # x-axis major label locations (Episode labels)
    #timestep_minor_locs = []  # x-axis minor label locations (Time step labels)

    for episode_idx in range_of_episodes:
        timestep_labels.append(episode_idx) # Episode
        act_val_length = len(actual_values)
        if act_val_length == 0:
            timestep_major_locs = [0]
        else:
            timestep_major_locs.append(act_val_length+1) # Timestep within the episode

        # Get the actual metric values within the current episode
        actual_values.extend(metrics_arr[episode_idx])

    fig, ax = plt.subplots()
    timesteps = np.arange(len(actual_values))
    # Plot average values
    plt.plot(timesteps, actual_values, label=metric_name)
    plt.title(str(metric_name) + " output for each episode")
    plt.xlabel("Episodes (with time steps)")
    plt.ylabel(str(metric_name))
    plt.legend(loc='best')
    plt.xlim([0, timesteps[-1]])
    plt.ylim([0., 0.8])

    #ax.xaxis.set_minor_locator(MultipleLocator(1))
    #ax.set_xticklabels(
    plt.xticks(timestep_major_locs, timestep_labels)
    #timestep_minor_locs = [x for x in timesteps if x not in timestep_major_locs]



    if saving_dir is not None:
        plt.savefig(saving_dir + "/" + metric_name + "act_vals_all_eps" + ".png")
        plt.close(fig)
    else:
        plt.show()

def episode_distribution_plot(metrics_arr):
    """ Plot distribution over one episode """
    plt.hist(metrics_arr)
    plt.title("Distribution over an episode")
    plt.show()


def all_episodes_average_plot(range_of_episodes, metrics_arr, metric_name, axes_limits, saving_dir=None):
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
    plt.plot(range_of_episodes, averages, label=metric_name)
    plt.title("Average value of "+ metric_name + " per episode")
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.legend(loc='best')
    plt.xlim([axes_limits['x_min'], axes_limits['x_max']])
    #plt.ylim([axes_limits['y_min'], axes_limits['y_max']])
    plt.ylim([0, 0.8])

    # Get general stats about metric output
    min_value, max_value, range_value, mean_value = get_stats(actual_values)

    # Return text string to be placed on plot (with stats about the metric)
    textstr = "\n{}:\n".format(metric_name)
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
        print("episode_idx: ",episode_idx)
        # Get all metrics from an episode
        episode_metric = metric_arr[episode_idx]

        metric_len = len(episode_metric[-1])

        # Convert each episode (list of transitions) into a stacked numpy array for manipulation
        episode_metric = np.stack(episode_metric, axis=0)

        # Get the specific metric index for each array
        if metric_len > 1:
            episode_metric = episode_metric[:, metric_idx]

        # Append the desired metric (at the metric_idx) from the current episode to a full list
        selected_episodes.append(episode_metric.tolist())

    return selected_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--replay_filepath", type=str, action='store', default=None) # Replay buffer filepath
    parser.add_argument("--metric_filepath", type=str, action='store', default=None) # Metric array filepath

    #parser.add_argument("--label_names", type=str, action='store', default="No Label") # List of labels for each plotted line (ex: Finger 1 Velocity)
    parser.add_argument("--metric_names", type=str, action='store', default=None) # Name of the metric being evaluated (ex: state, action, policy_action)
    parser.add_argument("--metric_indexes", type=str, action='store', default=None) # List of the indexes for each metric being evaluated (Ex: idx = 1, ReplayBuffer.action[idx] == Finger 1 Velocity)
    parser.add_argument("--start_episode_idx", type=int, action='store', default=0) # Starting index of a specific episode to evaluate
    parser.add_argument("--end_episode_idx", type=int, action='store', default=None)  # Ending index of a specific episode to evaluate

    parser.add_argument("--plot_type", type=str, action='store', default="actual") # Type of plot to be produced; Options: average, actual, distribution, boxplot
    parser.add_argument("--saving_dir", type=str, action='store', default=None) # Directory where plot output is saved
    parser.add_argument("--save_to_file", type=str, action='store', default="False") # Set to True to save the current metric to a text file

    args = parser.parse_args()

    #state_dim = 82
    #action_dim = 3
    #replay_size = 4000 # Size of the replay buffer (# Episodes)

    #replay_filepath = args.replay_filepath
    metric_filepath = args.metric_filepath

    metric_names = args.metric_names.split(',') # Name for each indexed metric (Ex: label_name for action[0] is "Finger 1 Velocity")
    #label_names = args.label_names.split(',') # Labels for each indexed metric (Ex: label_name for action[0] is "Finger 1 Velocity")
    metric_indexes = args.metric_indexes.split(',') # List of indexes for each metric

    start_episode_idx = args.start_episode_idx
    end_episode_idx = args.end_episode_idx

    plot_type = args.plot_type
    saving_dir = args.saving_dir
    if args.save_to_file == "True" or args.save_to_file == "true":
        save_to_file = True
    else:
        save_to_file = False


    #metric_arrays = []

    #if replay_filepath is not None:
    #    replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, replay_size) # Load the replay buffer (expert replay buffer)
    #    replay_text = replay_buffer.store_saved_data_into_replay(replay_filepath) # Load stored data into the initialized replay buffer
    #    replay_buffer_array = copy.deepcopy(replay_buffer.action)
    #    metric_arrays.append(replay_buffer_array)

    metric_array = []
    if metric_filepath is not None:
        metric_array = np.load(metric_filepath, allow_pickle=True)
        if len(metric_array[-1]) == 0:
            metric_array = metric_array[:-1]

    plt.figure() # Create the figure before plotting to allow multiple lines to be plotted
    textstr = "" # Text to be displayed on the plot (Ex: Min, Max, Avg. info per metric)

    if end_episode_idx is None or end_episode_idx > len(metric_array)-1:
        end_episode_idx = len(metric_array)-1

    # List of selected range of episode indexes
    range_of_episodes = np.arange(start_episode_idx, end_episode_idx+1) # Add 1 to include the final index

    # Holds the current metrics being analyzed within the selected episode range
    metric_dict = {"Episodes": range_of_episodes}

    for name, metric_idx in zip(metric_names, metric_indexes):
        # Get metrics from each of the desired episodes
        selected_metric_arr = get_selected_episode_metrics(range_of_episodes,metric_arr=metric_array, metric_idx=int(metric_idx))
        #timesteps = np.arrange(0,len(selected_metric_arr)+1)
        metric_dict[name] = selected_metric_arr

        if save_to_file is True and saving_dir is not None:
            df = pd.DataFrame(metric_dict)
            df.to_csv(saving_dir + "/" + name + ".csv")

        min_arr_value = min([min(metric_arr) for metric_arr in selected_metric_arr])
        max_arr_value = max([max(metric_arr) for metric_arr in selected_metric_arr])
        axes_limits = {"x_min": start_episode_idx, "x_max": end_episode_idx, "y_min": min_arr_value, "y_max": max_arr_value}

        # Plot the metrics based on the desired plot type
        if plot_type == "average":
            textstr += all_episodes_average_plot(range_of_episodes, selected_metric_arr, name, axes_limits, saving_dir)
        elif plot_type == "actual":
            actual_values_plot_all_episodes(range_of_episodes, selected_metric_arr, name, saving_dir)
        #elif plot_type == "distribution":
        #    all_episodes_distribution_plot(range_of_episodes, selected_metric_arr, metric_name)
        #elif plot_type == "boxplot":
        #    all_episodes_boxplot(range_of_episodes, selected_metric_arr, metric_name, freq=200, min_val=0, max_val=0.9)

        # Set matplotlib.patch.Patch properties
        props = dict(facecolor='white', pad=5, alpha=0.5)
        # Place a text box in the upper left corner
        plt.figtext(axes_limits["x_min"]+0.01, axes_limits["y_max"], textstr, fontsize=7,
                horizontalalignment='left', verticalalignment='top', bbox=props)
        plt.subplots_adjust(left=0.2)

        if saving_dir is None:
            plt.show()
        else:
            plt.savefig(saving_dir+"/"+name+plot_type+".png")