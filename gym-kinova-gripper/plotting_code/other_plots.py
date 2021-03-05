import matplotlib.pyplot as plt
import numpy as np

## Extra plotting functions that can be called for quick analysis

def plot_timestep_distribution(success_timesteps=None, fail_timesteps=None, all_timesteps=None, expert_saving_dir=None):
    """ Plot the distribution of time steps over successful and failed episodes """
    if all_timesteps is None:
        success_timesteps = np.load(expert_saving_dir + "/success_timesteps.npy")
        fail_timesteps = np.load(expert_saving_dir + "/fail_timesteps.npy")
        all_timesteps = np.load(expert_saving_dir + "/all_timesteps.npy")

    n_bins = 40
    # We can set the number of bins with the `bins` kwarg
    plt.hist(all_timesteps, bins=n_bins, color="g")
    plt.title("Total time steps distribution for all episodes (3x speed)", weight='bold')
    plt.xlabel('# of time steps per episode')
    plt.ylabel('# of episodes with the time step count')
    plt.xlim(0, 800)
    plt.savefig(expert_saving_dir + "/total_timestep_distribution")
    plt.clf()

    plt.hist(success_timesteps, bins=n_bins, color="b")
    plt.title("Time steps distribution for Successful episodes (3x speed)", weight='bold')
    plt.xlabel('# of time steps per episode')
    plt.ylabel('# of episodes with the time step count')
    plt.savefig(expert_saving_dir + "/success_timestep_distribution")
    plt.clf()

    plt.hist(fail_timesteps, bins=n_bins, color="r")
    plt.title("Time steps distribution for Failed episodes (3x speed)", weight='bold')
    plt.xlabel('# of time steps per episode')
    plt.ylabel('# of episodes with the time step count')
    plt.savefig(expert_saving_dir + "/fail_timestep_distribution")
    plt.clf()


'''
# Plot the average velocity over an episode
def plot_average_velocity(replay_buffer,num_timesteps):
    """ Plot the average velocity over a certain number of episodes """
    velocity_dir = "./expert_average_velocity"
    if not os.path.isdir(velocity_dir):
        os.mkdir(velocity_dir)

    #num_episodes = len(f1_vels)

    #plt.plot(np.arrange(len(f1_vels)), f1_vels)

    max_timesteps = 30
    timestep_vel_count = np.zeros(max_timesteps)
    wrist_avg_vels = np.zeros(max_timesteps)
    f1_avg_vels = np.zeros(max_timesteps)
    f2_avg_vels = np.zeros(max_timesteps)
    f3_avg_vels = np.zeros(max_timesteps)

    for episode_actions in replay_buffer.action:
        for timestep_idx in range(len(episode_actions)):
            timestep_vel_count[timestep_idx] += 1
            wrist_avg_vels[timestep_idx] = (wrist_avg_vels[timestep_idx] + episode_actions[timestep_idx][0]) / timestep_vel_count[timestep_idx]
            f1_avg_vels[timestep_idx] = (f1_avg_vels[timestep_idx] + episode_actions[timestep_idx][1]) / \
                                       timestep_vel_count[timestep_idx]
            f2_avg_vels[timestep_idx] = (f2_avg_vels[timestep_idx] + episode_actions[timestep_idx][2]) / \
                                       timestep_vel_count[timestep_idx]
            f3_avg_vels[timestep_idx] = (f3_avg_vels[timestep_idx] + episode_actions[timestep_idx][3]) / \
                                       timestep_vel_count[timestep_idx]

    num_episodes = len(replay_buffer.action)
    print("replay_buffer.action: ",replay_buffer.action)
    print("f1_avg_vels: ",f1_avg_vels)
    plt.plot(np.arange(num_timesteps), f1_avg_vels, color="r", label="Finger1")
    plt.plot(np.arange(num_timesteps), f2_avg_vels, color="b", label="Finger2")
    plt.plot(np.arange(num_timesteps), f3_avg_vels, color="g", label="Finger3")
    plt.plot(np.arange(num_timesteps), wrist_avg_vels, color="y", label="Wrist")
    plt.legend()

    plt.title("Average velocity over "+str(num_episodes)+" episodes", weight='bold')
    plt.xlabel('Timestep within an episode')
    plt.ylabel('Average Velocity at Timestep')
    #plt.savefig(velocity_dir + "/velocity_plot")
    #plt.clf()
    plt.show()
'''