import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

'''
    Generates boxplot from evaluation output, specifically avg. 
    reward data (finger, grasp, lift, total).
'''

def create_boxplot(orientation,saving_dir,data,labels,filename):
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
    if labels["freq_vals"] is not None:
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


def create_std_dev_bar_plot(x,y,deviation,labels,saving_dir,filename):
    """ Creates a plot that shows the input data points (x,y) and the deviation (deviation)
    x: x coordinate value (Episode)
    y: y coordinate value (Average Reward)
    deviation: Standard deviation per x,y point
    labels: Plot labels (Frequency x tick marks)
    saving_dir: Directory to save plot in
    filename: File name for the saved plot
    """
    plt.errorbar(x, y, deviation, linestyle='None', marker='^')

    plt.locator_params(axis='x', nbins=len(labels["freq_vals"]))
    locs, labl = plt.xticks()
    plt.xticks(locs, labels["freq_vals"])
    plt.title(labels["title"])

    if saving_dir is None:
        print("Showing standard deviation bar plot...")
        plt.show()
    else:
        plt.savefig(saving_dir+"/"+filename)
        print("Standard deviation bar plot saved at: ",saving_dir+"/"+filename)
    plt.show()

def get_boxplot_data(plot_type, data_dir,filename,tot_episodes,saving_freq):
    """Get boxplot data from saved numpy arrays
    data_dir: Directory location of data file
    filename: Name of data file
    tot_episodes: Total number of evaluation episodes
    saving_freq: Frequency in which the data files were saved
    """
    data = []
    if plot_type == "eval":
        for ep_num in np.linspace(start=saving_freq, stop=tot_episodes, num=int(tot_episodes/saving_freq), dtype=int):
            data_str = data_dir+filename+"_"+str(ep_num)+".npy"
            my_file = Path(data_str)
            if my_file.is_file():
                print("Eval file: ", data_str)
                data_file = np.load(data_str)[0]

                for eval_data in data_file:
                    data.append(eval_data)
            else:
                print("Boxplot file not found! file: ", data_str)
                data = None
    else:
        data_str = data_dir + filename + ".npy"
        my_file = Path(data_str)
        if my_file.is_file():
            print("Boxplot reward file: ", data_str)
            data_file = np.load(data_str)[0]

            for eval_data in data_file:
                data.append(eval_data)
        else:
            print("Boxplot file not found! file: ", data_str)
            data = None

    return data


def generate_reward_boxplots(plot_type, orientation, data_dir, saving_dir, file_list=["finger_reward", "grasp_reward", "lift_reward", "total_reward"], tot_episodes=20000, saving_freq=1000, eval_freq=200):
    """Create finger, grasp, lift, total reward evaluation boxplots
    data_dir: Directory location of data file
    saving_dir: Directory to save boxplots at
    file_list: Lift of data file names
    tot_episodes: Total number of evaluation episodes
    saving_freq: Frequency in which the data files were saved (i.e. 1000)
    eval_freq: Frequency in which the policy was evaluated (i.e. 200)
    """
    # Create saving directory if it does not exist
    plot_save_path = Path(saving_dir)
    plot_save_path.mkdir(parents=True, exist_ok=True)

    # Get each of the reward files
    for file in file_list:
        if plot_type == "eval":
            boxplot_data = get_boxplot_data(plot_type, data_dir, file, tot_episodes, saving_freq)

            freq_vals = np.arange(0,tot_episodes+1,200)

            avgs = []
            std_devs = []
            for row in boxplot_data:
                print("row: ",row)
                avgs.append(np.average(row))
                std_devs.append(np.std(row))
            x = np.arange(0, len(avgs))

            plot_labels = {"x_label": "Evaluation Episode", "y_label": "Total Avg. Reward",
                              "title": str(file)+" Avg. Reward per " + str(eval_freq) + " Grasp Trials, " + orientation + " Orientation", "freq_vals": freq_vals}

            if boxplot_data is not None:
                if len(boxplot_data) > 0:
                    create_boxplot(orientation,saving_dir, boxplot_data, plot_labels, "Eval_Boxplot_"+str(file)+".png")
                    create_std_dev_bar_plot(x, avgs, std_devs, plot_labels, saving_dir, "Eval_Std_Dev_plot_"+str(file)+".png")

        else:
            boxplot_data = get_boxplot_data(plot_type, data_dir, file, tot_episodes, saving_freq)
            freq_vals = np.arange(0, tot_episodes + 1, 200)
            plot_labels = {"x_label": "Episode", "y_label": "Total Avg. Reward",
                           "title": str(file) + " Avg. Reward" + orientation + " Orientation", "freq_vals": freq_vals}

            if boxplot_data is not None:
                create_boxplot(orientation, saving_dir, boxplot_data, plot_labels, "Boxplot_" + str(file) + ".png")


if __name__ ==  "__main__":
    # Code to test with
    data_dir = "./eval_plots/boxplot/"
    file_list = ["finger_reward", "grasp_reward", "lift_reward", "total_reward"]
    tot_episodes = 6000
    saving_freq = 1000
    eval_freq = 200
    saving_dir = "boxplot_output"
    orientation = "normal"

    output_dir = os.path.join(data_dir, saving_dir + "/")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Generates boxplots for each reward (finger, lift, grasp, total)
    generate_reward_boxplots(orientation, data_dir, saving_dir,file_list,tot_episodes, saving_freq, eval_freq)
