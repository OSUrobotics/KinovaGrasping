import matplotlib.pyplot as plt     # Plotting library

def reward_plot(axs, rewards, policy_name, label_name, color, episode_num):
    """ Plot the reward values from evaluation """
    axs.plot(rewards, label=label_name, color=color)
    axs.set_title("Episode {}".format(str(episode_num)))
    axs.set_xlabel("Grasp Trials")