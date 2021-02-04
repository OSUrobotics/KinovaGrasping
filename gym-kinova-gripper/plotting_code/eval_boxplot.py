import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

'''
    Generates boxplot from evaluation output, specifically avg. 
    reward data (finger, grasp, lift, total).
'''

def create_boxplot(saving_dir,data,labels,filename):
    """ Create a boxplot to show averages and spread of data
    saving_dir: Directory where plot will be saved
    x: x-axis data
    y: y-axis data
    labels: dictionary of plot labels in string format {"x_label":,"y_label":,"title":}
    filename: Name of file to save boxplot as
    """
    boxplot = sns.boxplot(data=data)
    #boxplot = sns.swarmplot(data=data) # Alternate boxplot style

    boxplot.set(xlabel=labels["x_label"], ylabel=labels["y_label"], title=labels["title"])
    plt.locator_params(axis='x', nbins=len(labels["freq_vals"]))
    locs, labl = plt.xticks()
    plt.xticks(locs, labels["freq_vals"])

    if saving_dir is None:
        print("Showing boxplot...")
        plt.show()
    else:
        plt.savefig(saving_dir+filename)
        print("Boxplot saved at: ",saving_dir+filename)

    # Clear current plot
    plt.clf()


def get_boxplot_data(data_dir,filename,tot_episodes,saving_freq):
    """Get boxplot data from saved numpy arrays
    data_dir: Directory location of data file
    filename: Name of data file
    tot_episodes: Total number of evaluation episodes
    saving_freq: Frequency in which the data files were saved
    """
    data = []
    for ep_num in np.linspace(start=saving_freq, stop=tot_episodes, num=int(tot_episodes/saving_freq), dtype=int):
        data_str = data_dir+filename+"_"+str(ep_num)+".npy"
        print("Eval file: ", data_str)
        data_file = np.load(data_str)[0]

        for eval_data in data_file:
            data.append(eval_data)

    return data


def generate_reward_boxplots(data_dir, saving_dir, file_list, tot_episodes, saving_freq, eval_freq):
    """Create finger, grasp, lift, total reward evaluation boxplots
    data_dir: Directory location of data file
    saving_dir: Directory to save boxplots at
    file_list: Lift of data file names
    tot_episodes: Total number of evaluation episodes
    saving_freq: Frequency in which the data files were saved (i.e. 1000)
    eval_freq: Frequency in which the policy was evaluated (i.e. 200)
    """
    for file in file_list:
        boxplot_data = get_boxplot_data(data_dir, file, tot_episodes, saving_freq)

        freq_vals = np.arange(0,tot_episodes,2000)

        boxplot_labels = {"x_label": "Evaluation Episode", "y_label": "Total Avg. Reward",
                          "title": str(file)+" Avg. Reward per " + str(eval_freq) + " Grasp Trials", "freq_vals": freq_vals}

        create_boxplot(saving_dir, boxplot_data, boxplot_labels, "Eval_Boxplot_"+str(file)+".png")


if __name__ ==  "__main__":
    # Code to test with
    data_dir = "./eval_plots/boxplot/"
    file_list = ["finger_reward", "grasp_reward", "lift_reward", "total_reward"]
    tot_episodes = 6000
    saving_freq = 1000
    eval_freq = 200
    saving_dir = "boxplot_output"

    output_dir = os.path.join(data_dir, saving_dir + "/")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Generates boxplots for each reward (finger, lift, grasp, total), saved at
    generate_reward_boxplots(data_dir, output_dir, file_list, tot_episodes, saving_freq, eval_freq)
