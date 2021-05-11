import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.patches import Rectangle
import utils
import argparse
import os, sys
expert_path = os.getcwd()
sys.path.insert(1, expert_path)
#from expert_data import GenerateExpertPID_JointVel


def plot_finger(ax, finger_x, finger_y, actions, markers_on, ts_spacing, action_label_freq, scatter_color, marker_type, marker_color, label):
    """ Plot a particular finger trajectory """
    # Plot the course of the trajectory through a thin line
    plt.plot(finger_x[-1], finger_y[-1], marker="X", ms=4, linewidth=1, color='r')
    im = plt.scatter(finger_x, finger_y, c=ts_spacing, marker=marker_type, cmap=scatter_color, label=label)
    # Plot the final position
    plt.plot(finger_x, finger_y, ms=4, linewidth=1, color=marker_color, markevery=markers_on)
    bbox_props = dict(boxstyle='square,pad=1', fc='none', ec='none')
    for i, action_txt in enumerate(actions):
        if action_label_freq is not None:
            if i % action_label_freq == 0:
                ax.annotate("TS {}) F1: {} F2: {} F3: {}".format(i,action_txt[0],action_txt[1],action_txt[2]), (finger_x[i], finger_y[i]), bbox=bbox_props)

def get_palm_coordinates(states):
    """ Get each of the palm coordinate values for each state within the one episode
    states: List of each state recorded within a single episode (local coordinates wrt. the center of the palm)
    """
    palm_x = [ts_state[18] for ts_state in states]
    palm_y = [ts_state[19] for ts_state in states]
    palm_z = [ts_state[20] for ts_state in states]

    return palm_x, palm_y, palm_z

def get_proximal_finger_coordinates(states):
    """ Get each of the proximal finger postion values for each state within the one episode
    states: List of each state recorded within a single episode (local coordinates wrt. the center of the palm)
    """
    # Finger distal x,y coordinates
    # State indexes: (9 - 11) "f1_dist", (12 - 14) "f2_dist", (15 - 17) "f3_dist"
    f1_prox_x = [ts_state[0] for ts_state in states] # Finger 1
    f1_prox_y = [ts_state[1] for ts_state in states]
    f2_prox_x = [ts_state[3] for ts_state in states] # Finger 2
    f2_prox_y = [ts_state[4] for ts_state in states]
    f3_prox_x = [ts_state[6] for ts_state in states] # Finger 3
    f3_prox_y = [ts_state[7] for ts_state in states]

    return f1_prox_x, f1_prox_y, f2_prox_x, f2_prox_y, f3_prox_x, f3_prox_y

def get_distal_finger_coordinates(states):
    """ Get each of the distal finger postion values for each state within the one episode
    states: List of each state recorded within a single episode (local coordinates wrt. the center of the palm)
    """
    # Finger distal x,y coordinates
    # State indexes: (9 - 11) "f1_dist", (12 - 14) "f2_dist", (15 - 17) "f3_dist"
    f1_dist_x = [ts_state[9] for ts_state in states] # Finger 1
    f1_dist_y = [ts_state[10] for ts_state in states]
    f2_dist_x = [ts_state[12] for ts_state in states] # Finger 2
    f2_dist_y = [ts_state[13] for ts_state in states]
    f3_dist_x = [ts_state[15] for ts_state in states] # Finger 3
    f3_dist_y = [ts_state[16] for ts_state in states]

    print("Finger1 distal, Finger2 distal, Finger3 distal x,y: {},{}  {},{}  {},{}".format(f1_dist_x[0],f1_dist_y[0],f2_dist_x[0],f2_dist_y[0],f3_dist_x[0],f3_dist_y[0]))

    return f1_dist_x, f1_dist_y, f2_dist_x, f2_dist_y, f3_dist_x, f3_dist_y

def get_object_coordinates(states):
    """ Get each of the object postion (x,y,z) for each state within the one episode
    states: List of each state recorded within a single episode (local coordinates wrt. the center of the palm)
    """
    # Object coordinates
    object_x = [ts_state[21] for ts_state in states]    # Object x-coordinates
    object_y = [ts_state[22] for ts_state in states]    # Object y-coordinates
    object_z = [ts_state[23] for ts_state in states]    # Object z-coordinates

    return object_x, object_y, object_z

def plot_trajectory(states, actions, episode_num, saving_dir):
    """ Plot the trajectory of the fingers and object over an episode
    image: Image of the simulation at the final point
    states: List of each state array recorded within a single episode
    actions: Finger velocities over the course of an episode
    object_coords: x,y coordinates of the object throughout the episode
    """
    # Get finger and object coordinates from the episode
    palm_x, palm_y, palm_z = get_palm_coordinates(states)
    f1_dist_x, f1_dist_y, f2_dist_x, f2_dist_y, f3_dist_x, f3_dist_y = get_distal_finger_coordinates(states)
    f1_prox_x, f1_prox_y, f2_prox_x, f2_prox_y, f3_prox_x, f3_prox_y = get_proximal_finger_coordinates(states)
    object_x, object_y, object_z = get_object_coordinates(states)

    ts_spacing = np.arange(start=0, stop=len(object_x))    # Just used to determine scatter plot spacing

    # Min/Max values determined based on the range of object positions within the hand
    x_min = -0.09
    x_max = 0.09
    y_min = 0
    y_max = 0.07
    x_range = x_max - (x_min)
    y_range = y_max - y_min
    x_bins = int(x_range / 0.002)
    y_bins = int(y_range / 0.002)

    # Object sizes for drawing box around object coordinate
    # Small half-size width 0.0175 (full width: 0.035), half-size height 0.05 (full height: 0.1)
    obj_width = 0.035
    obj_height = 0.035

    # Create figure with axis to create more refined axis edits
    fig, ax = plt.subplots()

    # Axis limit and ticks determined by min/max coordinate values and even spacing
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.yticks(np.arange(x_min, x_max, x_bins))
    plt.xticks(np.arange(y_min, y_max, y_bins))

    # Adjust axis ticks based on size of plot
    ax.set_aspect('equal', adjustable='box')  # Ensure axis tick spacing is equal
    fig.set_size_inches(11, 7)  # Set figure size to make a larger depiction of the plot than default
    ax.xaxis.set_major_locator(MultipleLocator(0.01))  # Set axis tick locations
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))

    # Plot labels
    plot_title = "Naive Controller: Small Cube, Normal Orienation\nObject coordinates and finger velocities over episode set"
    plt.title(plot_title)
    plt.xlabel('X-axis coordinate position (meters)')
    plt.ylabel('Y-axis coordinate position (meters)')

    markers_on = [0, 5, 10, 15, 20]
    action_label_freq = 5 # Frequency to label action output

    # Plot FINGER 1 (x,y) position over the course of the episode
    plot_finger(ax, f1_dist_x, f1_dist_y, actions, markers_on, ts_spacing, action_label_freq, scatter_color='Purples', marker_type=">", marker_color='#8a5ab8', label="Finger 1")
    # Plot FINGER 2 (x,y) position over the course of the episode
    plot_finger(ax, f2_dist_x, f2_dist_y, actions, markers_on, ts_spacing, action_label_freq, scatter_color='Blues', marker_type="<", marker_color='#8a5ab8', label="Finger 2")
    # Plot FINGER 3 (x,y) position over the course of the episode
    plot_finger(ax, f3_dist_x, f3_dist_y, actions, markers_on, ts_spacing, action_label_freq=None, scatter_color='Oranges', marker_type='D', marker_color='#8a5ab8', label="Finger 3")

    # Plot object (x,y) position over the course of the episode
    plt.plot(object_x[-1], object_y[-1], marker="X", ms=4, linewidth=1, color='r')
    im = plt.scatter(object_x, object_y, c=ts_spacing, cmap='viridis', label="Object")
    ax.annotate("Object Initial Coord (x,y): {:.2f},{:.2f}".format(object_x[0], object_x[0]), (object_x[0], object_y[0]))

    # Set legend based on plotted trajectories
    # plt.legend(loc="upper left")

    # Create color bar showing the time step within an episode
    cb = plt.colorbar(cmap='viridis', shrink=0.6, ticks=[0, 5, 10, 15, 20, 25, 30])
    plt.clim(0, 30)
    cb.ax.set_yticklabels(['0', '5', '10', '15', '20', '25', '30'])
    cb_label = "Time step within episode"
    cb.set_label(cb_label)

    """
    # Add box around the final object location to show the height/width
    for a_x, a_y in zip(*([object_x[-1]], [object_y[-1]])):
        ax.add_patch(Rectangle(
            xy=(a_x - obj_width / 2, a_y - obj_height / 2), width=obj_width, height=obj_height,
            linewidth=1, color='blue', fill=False))

    # Add box around the final FINGER 1 DISTAL location to show the height/width
    f_dist_width = 0.029
    f_dist_height = 0.044
    for a_x, a_y in zip(*([f1_x[-1]], [f1_y[-1]])):
        ax.add_patch(Rectangle(
            xy=(a_x - f_dist_width / 2, a_y - f_dist_height / 2), width=f_dist_width, height=f_dist_height,
            linewidth=1, color='red', fill=False))
    """

    if saving_dir is None:
        plt.show()
    else:
        fig_filename = "traj_plot_ep_"+str(episode_num)
        plt.savefig(saving_dir + fig_filename)
        plt.clf()


if __name__ == "__main__":
    print("--- Plot the finger and object trajectory over a set of episodes ---")
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_filepath", type=str, action='store', default=None)  # Replay buffer filepath
    args = parser.parse_args()

    replay_filepath = args.replay_filepath

    pid_mode = "naive_only"
    replay_size = 10    # Number of episodes to generate in replay buffer
    with_grasp = False
    test_ep_num = 0     # Episode number to test plotting with from replay buffer
    fig_filename = "Trajectory Plot Test"
    saving_dir = None   # If None, plot_trajectory will show the plot instead of saving
    requested_shapes = "CubeS"
    requested_orientation = "normal"

    # Generate temporary replay buffer to test trajectory plotting with
    replay_buffer = utils.ReplayBuffer_Queue(state_dim=82, action_dim=4)
    if replay_filepath is None:
        replay_buffer, replay_filepath, expert_data_dir, info_file_text, num_success, num_total, coord_filepath = GenerateExpertPID_JointVel(replay_size, requested_shapes, requested_orientation, replay_buffer=replay_buffer, with_grasp=False, with_noise=False, save=True, render_imgs=False, pid_mode="naive")
    else:
        replay_text = replay_buffer.store_saved_data_into_replay(replay_filepath)

    # Plot finger and object trajectory over the course of the desired episode(s)
    plot_trajectory(replay_buffer.state[test_ep_num], replay_buffer.actions[test_ep_num], fig_filename, saving_dir)