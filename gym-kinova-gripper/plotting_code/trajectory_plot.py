import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.patches import Rectangle
import utils
from expert_data import GenerateExpertPID_JointVel

def plot_trajectory(states, actions, episode_num, saving_dir):
    """ Plot the trajectory of the fingers and object over an episode
    image: Image of the simulation at the final point
    actions: Finger velocities over the course of an episode
    object_coords: x,y coordinates of the object throughout the episode
    """
    # Finger distal x,y coordinates
    # State indexes: (9 - 11) "f1_dist", (12 - 14) "f2_dist", (15 - 17) "f3_dist"
    f1_x = [ts_state[9] for ts_state in states] # Finger 1
    f1_y = [ts_state[10] for ts_state in states]
    f2_x = [ts_state[12] for ts_state in states] # Finger 2
    f2_y = [ts_state[13] for ts_state in states]
    f3_x = [ts_state[15] for ts_state in states] # Finger 3
    f3_y = [ts_state[16] for ts_state in states]
    
    # Object coordinates
    object_x = [ts_state[21] for ts_state in states]    # Object x-coordinates
    object_y = [ts_state[22] for ts_state in states]    # Object y-coordinates
    object_z = [ts_state[23] for ts_state in states]    # Object z-coordinates
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
    plot_title = "Object coordinates and finger velocities over episode set"
    plt.title(plot_title)
    plt.xlabel('X-axis coordinate position (meters)')
    plt.ylabel('Y-axis coordinate position (meters)')

    markers_on = [0, 5, 10, 15, 20]

    # Plot FINGER 1 (x,y) position over the course of the episode
    plt.plot(f1_x[-1], f1_y[-1], marker="X", ms=4, linewidth=1, color='r')
    im = plt.scatter(f1_x, f1_y, c=ts_spacing, cmap='Purples')
    plt.plot(f1_x, f1_y, ms=4, linewidth=1, color='#8a5ab8', markevery=markers_on)

    # Plot FINGER 2 (x,y) position over the course of the episode
    plt.plot(f2_x[-1], f2_y[-1], marker="X", ms=4, linewidth=1, color='r')
    im = plt.scatter(f2_x, f2_y, c=ts_spacing, cmap='Blues')
    plt.plot(f2_x, f2_y, ms=4, linewidth=1, color='#8a5ab8', markevery=markers_on)

    # Plot FINGER 3 (x,y) position over the course of the episode
    plt.plot(f3_x[-1], f3_y[-1], marker="X", ms=4, linewidth=1, color='r')
    im = plt.scatter(f3_x, f3_y, c=ts_spacing, cmap='Oranges')
    plt.plot(f3_x, f3_y, ms=4, linewidth=1, color='#8a5ab8', markevery=markers_on)

    # Plot object (x,y) position over the course of the episode
    plt.plot(object_x[-1], object_y[-1], marker="X", ms=4, linewidth=1, color='r')
    im = plt.scatter(object_x, object_y, c=ts_spacing, cmap='viridis')

    # Create color bar showing the time step within an episode
    cb = plt.colorbar(cmap='viridis', shrink=0.6, ticks=[0, 5, 10, 15, 20, 25, 30])
    plt.clim(0, 30)
    cb.ax.set_yticklabels(['0', '5', '10', '15', '20', '25', '30'])
    cb_label = "Time step within episode"
    cb.set_label(cb_label)

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

    if saving_dir is None:
        plt.show()
    else:
        fig_filename = "traj_plot_ep_"+str(episode_num)
        plt.savefig(saving_dir + fig_filename)
        plt.clf()


if __name__ == "__main__":
    print("--- Plot the finger and object trajectory over a set of episodes ---")

    pid_mode = "naive_only"
    replay_size = 10    # Number of episodes to generate in replay buffer
    with_grasp = False
    test_ep_num = 0     # Episode number to test plotting with from replay buffer
    fig_filename = "Trajectory Plot Test"
    saving_dir = None   # If None, plot_trajectory will show the plot instead of saving

    # Generate temporary replay buffer to test trajectory plotting with
    replay_buffer = utils.ReplayBuffer_Queue(state_dim=82, action_dim=4)
    replay_buffer, save_filepath, expert_saving_dir, text, num_success, total = GenerateExpertPID_JointVel(episode_num=replay_size, requested_shapes=["CubeS"], requested_orientation="normal", with_grasp=with_grasp, replay_buffer=replay_buffer, save=False, render_imgs=True, pid_mode=pid_mode)

    # Plot finger and object trajectory over the course of the desired episode(s)
    plot_trajectory(replay_buffer.state[test_ep_num], replay_buffer.actions[test_ep_num], fig_filename, saving_dir)