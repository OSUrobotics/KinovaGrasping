import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # Ignore divide by 0 warnings as we handle for it
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from PIL import Image
import os
import argparse
from pathlib import Path

def heatmap_freq(total_x,total_y,plot_title,fig_filename,saving_dir):
    """ Create heatmap displaying frequency of object initial starting position coordinates
    total_x: Total initial object position x-coordinates
    total_y: Total initial object position y-coordinates
    plot_title, plot_name, fig_filename, saving_dir
    """
    title = plot_title
    cb_label = 'Frequency count of all grasp trials'
    x_range = 0.09 - (-0.09)
    y_range = 0.07
    x_bins = int(x_range / 0.002)
    y_bins = int(y_range / 0.002)

    # Get total coordinates within their respective bins
    total_data, x_edges, y_edges = np.histogram2d(total_x,total_y, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

    fig, ax = plt.subplots()

    # Plot heatmap from data
    im = ax.imshow(total_data.T, cmap=plt.cm.Oranges, interpolation='none', origin='lower',extent=[-.09, 0.09, 0, 0.07])
    ax.set_aspect('equal', adjustable='box')

    fig.set_size_inches(11,8)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))   # Set axis tick locations
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))

    # Plot labels
    plt.title(plot_title)
    plt.xlabel('X-axis initial coordinate position of object (meters)')
    plt.ylabel('Y-axis initial coordinate position of object (meters)')

    # Create color bar showing overall frequency count values
    cb = plt.colorbar(mappable=im,shrink=0.6)
    cb.set_label(cb_label)

    #plt.show()
    plt.savefig(saving_dir+fig_filename)
    plt.clf()

def heatmap_plot(success_x,success_y,fail_x,fail_y,total_x,total_y,plot_title,fig_filename,saving_dir,plot_success):
    """ Create heatmap displaying success rate of object initial position coordinates """
    cb_label = 'Success rate of grasp trials out of total trials (Negative is failure rate)'
    x_range = 0.09 - (-0.09)
    y_range = 0.07
    x_bins = int(x_range / 0.002)
    y_bins = int(y_range / 0.002)

    # Get success/fail coordinates within their respective bins
    success_data, _, _ = np.histogram2d(success_x, success_y, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))
    fail_data, _, _ = np.histogram2d(fail_x, fail_y, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))
    total_data, x_edges, y_edges = np.histogram2d(total_x, total_y, range=[[-.09, 0.09], [0, 0.07]],bins=(x_bins,y_bins))

    # Positive Success rate coordinates bins
    pos_success_data = np.divide(success_data,total_data)
    pos_success_data = np.nan_to_num(pos_success_data)

    # Negative Success rate (Failed) coordinates bins
    neg_success_data = np.divide(fail_data,total_data)
    neg_success_data = np.nan_to_num(neg_success_data)
    neg_success_data = np.multiply(-1,neg_success_data)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot heatmap from data
    if plot_success is True:
        plt.imshow(pos_success_data.T, cmap=plt.cm.RdBu,  origin='lower',extent=[-.09, 0.09, 0, 0.07],vmin=-1, vmax=1)
    else:
        plt.imshow(neg_success_data.T, cmap=plt.cm.RdBu, origin='lower', extent=[-.09, 0.09, 0, 0.07], vmin=-1, vmax=1)

    ax.set_aspect('equal', adjustable='box')    # Sets histogram bin format
    fig.set_size_inches(11,8)   # Figure size
    ax.xaxis.set_major_locator(MultipleLocator(0.01))   # Set axis tick locations
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))

    # Plot labels
    plt.title(plot_title)
    plt.xlabel('X-axis initial coordinate position of object (meters)')
    plt.ylabel('Y-axis initial coordinate position of object (meters)')

    # Create color bar with success rate percent labels (0 to 100% success)
    cb = plt.colorbar(cmap=plt.cm.RdBu, format=PercentFormatter(1), shrink=0.6, ticks=[-1, -.5, 0, .5, 1])
    cb.ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    cb.set_label(cb_label)

    plt.savefig(saving_dir+fig_filename)
    #plt.show()
    plt.clf()


def make_img_transparent(fig_filename,saving_dir):
    """ Make plot image background transparent for overlaying on other plots
    img_name: Name of image to be made transparent
    dir: Directory to get plot from
    """
    # Get the image as an rgba array
    img = Image.open(saving_dir+fig_filename+".png")
    img = img.convert("RGBA")
    datas = img.getdata()

    # Convert rgb values that are white (225,225,225) to an alpha of 0 (transparent)
    newData = []
    for item in datas:
      if item[0] == 255 and item[1] == 255 and item[2] == 255:
          newData.append((255, 255, 255, 0))
      elif item[0] == 255 and item[1] == 253 and item[2] == 253:
          newData.append((255, 255, 255, 0))
      else:
          newData.append(item)

    # Save new transparent plot in original location
    img.putdata(newData)
    img.save(saving_dir+fig_filename+".png", "PNG",transparent=True)


def create_heatmaps(success_x,success_y,fail_x,fail_y,total_x,total_y,ep_str,orientation,saving_dir):
    """ Calls ferquency and success/fail heatmap plots 
    success_x: Successful initial object position x-coordinates
    success_y: Successful initial object position y-coordinates
    fail_x: Fail initial object position x-coordinates
    fail_y: Fail initial object position y-coordinates
    total_x: Total initial object position x-coordinates
    total_y: Total initial object position y-coordinates
    saving_dir: Directory to save plot output to
    """
    # Create frequency and success_rate heatmap plot directories
    heatmap_saving_dir = saving_dir + "/heatmap_plots/"
    if not os.path.isdir(heatmap_saving_dir):
        os.mkdir(heatmap_saving_dir)

    freq_saving_dir = saving_dir + "/freq_plots/"
    if not os.path.isdir(freq_saving_dir):
        os.mkdir(freq_saving_dir)

    title_str = ""
    if ep_str != "":
        title_str = ", Evaluated at Ep. " + ep_str
        ep_str = "_" + ep_str

    title_str += " " + orientation + " Orientation"

    # Plot frequency heatmap
    freq_plot_title = "Grasp Trial Frequency per Initial Pose of Object" + title_str
    heatmap_freq(total_x,total_y,freq_plot_title,'freq_heatmap'+ep_str+'.png',freq_saving_dir)

    # Plot failed (negative success rate) heatmap
    fail_plot_title = "Grasp Trial Success Rate per Initial Coordinate Position of Object" + title_str
    heatmap_plot(success_x, success_y, fail_x, fail_y, total_x, total_y, fail_plot_title, 'fail_heatmap'+ep_str+'.png',
                 heatmap_saving_dir, plot_success=False)

    # Plot successful heatmap
    success_plot_title = "Grasp Trial Success Rate per Initial Coordinate Position of Object" + title_str
    heatmap_plot(success_x, success_y, fail_x, fail_y, total_x, total_y,
                             success_plot_title, 'success_heatmap'+ep_str+'.png', heatmap_saving_dir,plot_success=True)

    # Make successful heatmap transparent to overlay over failed heatmap
    fig_filename = 'success_heatmap'+ep_str
    make_img_transparent(fig_filename,heatmap_saving_dir)

def get_heatmap_coord_data(data_dir,ep_str):
    """Get boxplot data from saved numpy arrays
    data_dir: Directory location of data file
    filename: Name of data file
    tot_episodes: Total number of evaluation episodes
    saving_freq: Frequency in which the data files were saved
    """
    success_x = np.load(data_dir+"success_x"+ep_str+".npy")
    success_y = np.load(data_dir+"success_y"+ep_str+".npy")
    fail_x = np.load(data_dir+"fail_x"+ep_str+".npy")
    fail_y = np.load(data_dir+"fail_y"+ep_str+".npy")
    total_x = np.load(data_dir+"total_x"+ep_str+".npy")
    total_y = np.load(data_dir+"total_y"+ep_str+".npy")

    return success_x, success_y, fail_x, fail_y, total_x, total_y


def generate_heatmaps(plot_type, orientation, data_dir, saving_dir, saving_freq=1000, tot_episodes=20000):
    """ Controls whether train or evaluation heatmap plots are generated
    plot_type: Type of plot (train or eval - eval will produce multiple plots)
    data_dir: Directory location of data file (coordinates)
    saving_dir: Directory to save the heatmap plots to
    saving_freq: Frequency at which the data was saved (used for getting filenames)
    tot_episodes: Total number of episodes the data covers (to get up to the last data file)
    """
    # If input data directory is not found, return back
    if not os.path.isdir(data_dir):
        return

    # Create saving directory if it does not exist
    plot_save_path = Path(saving_dir)
    plot_save_path.mkdir(parents=True, exist_ok=True)

    # FOR EVAL
    if plot_type == "eval":
        for ep_num in np.linspace(start=saving_freq, stop=tot_episodes, num=int(tot_episodes / saving_freq), dtype=int):
            print("EVAL EP NUM: ",ep_num)
            ep_str = str(ep_num)
            # Get coordinate data as numpy arrays
            success_x, success_y, fail_x, fail_y, total_x, total_y = get_heatmap_coord_data(data_dir, "_"+ep_str)
            #print("success_x, success_y: (",success_x,", ",success_y,")")
            # Plot coordinate data to frequency and success rate heatmaps
            create_heatmaps(success_x, success_y, fail_x, fail_y, total_x, total_y, ep_str, orientation, saving_dir)
    else:
        # FOR TRAIN
        ep_str = ""
        # Get coordinate data as numpy arrays
        success_x, success_y, fail_x, fail_y, total_x, total_y = get_heatmap_coord_data(data_dir, ep_str)
        # Plot coordinate data to frequency and success rate heatmaps
        create_heatmaps(success_x, success_y, fail_x, fail_y, total_x, total_y, ep_str, orientation, saving_dir)

    print(plot_type + "plots saved at: ", saving_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, action='store', default="./")
    parser.add_argument("--saving_dir", type=str, action='store', default="./")
    parser.add_argument("--plot_type", type=str, action='store', default="train")
    parser.add_argument("--orientation", type=str, action='store', default="normal")
    args = parser.parse_args()

    data_dir = args.data_dir       # Directory to Get input data from
    saving_dir = args.saving_dir   # Directory to Save data to
    plot_type = args.plot_type     # Train or Eval plots
    orientation = args.orientation # Hand orientation

    if data_dir[-1] != "/":
        data_dir += "/"

    print("Getting heatmap plots data from: ",data_dir)
    print("Saving heatmap plots at: ",saving_dir)

    generate_heatmaps(plot_type, data_dir, orientation, saving_dir)