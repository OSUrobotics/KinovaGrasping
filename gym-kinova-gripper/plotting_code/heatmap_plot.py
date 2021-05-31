import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # Ignore divide by 0 warnings as we handle for it
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from PIL import Image
import os
import argparse
from pathlib import Path

def heatmap_actual_coords(total_x,total_y,hand_lines,state_rep,plot_title,fig_filename,saving_dir):
    """ Create heatmap displaying the actual locations of object initial starting position coordinates
    total_x: Total initial object position x-coordinates
    total_y: Total initial object position y-coordinates
    plot_title, plot_name, fig_filename, saving_dir
    """
    title = plot_title
    cb_label = 'Frequency count of all grasp trials'
    x_min = -0.12
    x_max = 0.12
    y_min = 0
    y_max = 0.09

    x_range = x_max - (x_min)
    y_range = y_max - y_min
    x_bins = int(x_range / 0.002)
    y_bins = int(y_range / 0.002)

    if state_rep == "global":
        color_value = '#88c999'  # Green
        label = "Global"
        loc = 'upper right'
    elif state_rep == "local":
        color_value = '#a64dff'  # Purple
        label = "Local"
        loc = 'lower right'
    elif "local_to_global":
        color_value = "#ff4d88" # Pink
        label = "Local --> Global"
        loc = 'upper left'

    fig = plt.figure()
    fig.set_size_inches(11,8)   # Figure size
    fig.set_dpi(100)           # Pixel amount
    ax = fig.add_subplot(111)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.scatter(total_x,total_y, label = label, color = color_value)

    # Plot the position of each of the fingers on top of the heatmap plot
    plt.plot(hand_lines["finger1"][0],hand_lines["finger1"][1], label="Finger 1")
    plt.plot(hand_lines["finger2"][0],hand_lines["finger2"][1], label="Finger 2")
    plt.plot(hand_lines["finger3"][0],hand_lines["finger3"][1], label="Finger 3")

    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MultipleLocator(0.01))   # Set axis tick locations
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))

    # Add the legend
    ax.legend(title="Coordinate Frame", loc=loc)

    # Plot labels
    plt.title(plot_title)
    plt.xlabel('X-axis coordinate position (meters)')
    plt.ylabel('Y-axis coordinate position (meters)')

    if saving_dir is None:
        plt.show()
    else:
        plt.savefig(saving_dir+fig_filename)
        plt.close(fig)

def heatmap_freq(total_x,total_y,hand_lines,state_rep,plot_title,fig_filename,saving_dir):
    """ Create heatmap displaying frequency of object initial starting position coordinates
    total_x: Total initial object position x-coordinates
    total_y: Total initial object position y-coordinates
    plot_title, plot_name, fig_filename, saving_dir
    """
    title = plot_title
    cb_label = 'Frequency count of all grasp trials'
    x_min = -0.12
    x_max = 0.12
    y_min = 0
    y_max = 0.09

    x_range = x_max - (x_min)
    y_range = y_max - y_min
    x_bins = int(x_range / 0.002)
    y_bins = int(y_range / 0.002)

    # Get total coordinates within their respective bins
    total_data, x_edges, y_edges = np.histogram2d(total_x,total_y, range=[[x_min, x_max], [y_min, y_max]],bins=(x_bins,y_bins))

    fig = plt.figure()
    fig.set_size_inches(11,8)   # Figure size
    fig.set_dpi(100)           # Pixel amount
    ax = fig.add_subplot(111)

    # Plot heatmap from data
    im = ax.imshow(total_data.T, cmap=plt.cm.Purples, interpolation='none', origin='lower',extent=[x_min, x_max, y_min, y_max])

    # Plot the position of each of the fingers on top of the heatmap plot
    plt.plot(hand_lines["finger1"][0],hand_lines["finger1"][1], label="Finger 1")
    plt.plot(hand_lines["finger2"][0],hand_lines["finger2"][1], label="Finger 2")
    plt.plot(hand_lines["finger3"][0],hand_lines["finger3"][1], label="Finger 3")

    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MultipleLocator(0.01))   # Set axis tick locations
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))

    # Plot labels
    plt.title(plot_title)
    plt.xlabel('X-axis coordinate position (meters)')
    plt.ylabel('Y-axis coordinate position (meters)')

    # Create color bar showing overall frequency count values
    cb = plt.colorbar(mappable=im,shrink=0.6)
    cb.set_label(cb_label)

    if saving_dir is None:
        plt.show()
    else:
        plt.savefig(saving_dir+fig_filename)
        plt.close(fig)

def heatmap_plot(success_x,success_y,fail_x,fail_y,total_x,total_y,hand_lines,state_rep,plot_title,fig_filename,saving_dir,plot_success):
    """ Create heatmap displaying success rate of object initial position coordinates """
    cb_label = 'Grasp Trial Success Rate %'

    x_min = -0.12
    x_max = 0.12
    y_min = 0
    y_max = 0.09

    x_range = x_max - (x_min)
    y_range = y_max - y_min
    x_bins = int(x_range / 0.002)
    y_bins = int(y_range / 0.002)

    # Get success/fail coordinates within their respective bins
    success_data, _, _ = np.histogram2d(success_x, success_y, range=[[x_min, x_max], [y_min, y_max]],bins=(x_bins,y_bins))
    fail_data, _, _ = np.histogram2d(fail_x, fail_y, range=[[x_min, x_max], [y_min, y_max]],bins=(x_bins,y_bins))
    total_data, x_edges, y_edges = np.histogram2d(total_x, total_y, range=[[x_min, x_max], [y_min, y_max]],bins=(x_bins,y_bins))

    """ Testing plotting
    print("\nPlotting: saving_dir+fig_filename",saving_dir+fig_filename)
    print("plot_success: ",plot_success)
    print("success_x: ",success_x)
    print("success_y: ", success_y)
    print("fail_x: ", fail_x)
    print("fail_y: ", fail_y)
    print("x_bins: ",x_bins)
    print("y_bins: ", y_bins)
    print("len(fail_x): ",len(fail_x))
    print("sum(fail_data): ",np.sum(fail_data))
    print("sum(success_data): ", np.sum(success_data))
    print("total_x: ", total_x)
    #print("success_data: ",success_data)
    #print("sum(success_data): ",sum(success_data))
    print("len(success_x): ",len(success_x))
    """

    # Positive Success rate coordinates bins
    if len(fail_x) == 0:
        pos_success_data = total_data #success_data
    else:
        pos_success_data = np.divide(success_data,total_data)
        pos_success_data = np.nan_to_num(pos_success_data)

    # Negative Success rate (Failed) coordinates bins
    if len(success_x) == 0:
        neg_success_data = total_data #fail_data
    else:
        neg_success_data = np.divide(fail_data,total_data)
        neg_success_data = np.nan_to_num(neg_success_data)
    neg_success_data = np.multiply(-1,neg_success_data)

    fig = plt.figure()
    fig.set_size_inches(11,8)   # Figure size
    fig.set_dpi(100)           # Pixel amount
    ax = fig.add_subplot(111)

    # Plot heatmap from data
    if plot_success is True:
        plt.imshow(pos_success_data.T, cmap=plt.cm.RdBu,  origin='lower',extent=[x_min, x_max, y_min, y_max],vmin=-1, vmax=1)
    else:
        plt.imshow(neg_success_data.T, cmap=plt.cm.RdBu, origin='lower', extent=[x_min, x_max, y_min, y_max], vmin=-1, vmax=1)

    # Plot the position of each of the fingers on top of the heatmap plot
    plt.plot(hand_lines["finger1"][0],hand_lines["finger1"][1], label="Finger 1")
    plt.plot(hand_lines["finger2"][0],hand_lines["finger2"][1], label="Finger 2")
    plt.plot(hand_lines["finger3"][0],hand_lines["finger3"][1], label="Finger 3")

    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MultipleLocator(0.01))   # Set axis tick locations
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))

    # Plot labels
    plt.title(plot_title)
    plt.xlabel('X-axis coordinate position (meters)')
    plt.ylabel('Y-axis coordinate position (meters)')
    plt.legend() # Plot the legend (displays each finger line)

    # Create color bar with success rate percent labels (0 to 100% success)
    cb = plt.colorbar(cmap=plt.cm.RdBu, format=PercentFormatter(1), shrink=0.6, ticks=[-1, -.5, 0, .5, 1])
    cb.ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    cb.set_label(cb_label)

    plt.savefig(saving_dir+fig_filename)
    plt.close()
    #plt.show()


def change_image_transparency(image_filepath,alpha=0):
    """ Make plot image background transparent for overlaying on other plots
    img_name: Name of image to be made transparent
    dir: Directory to get plot from
    alpha: Changes the transparency [0,1], where 0 is fully transparent, 1 is opaque
    """
    # Get the image as an rgba array
    img = Image.open(image_filepath)
    img = img.convert("RGBA")
    datas = img.getdata()

    # Convert rgb values that are white (225,225,225) to an alpha of 0 (transparent)
    newData = []
    for item in datas:
        if item[3] == 0 and alpha > 0:
            alpha_value = int(alpha*240)
            newData.append((250, 250, 250, alpha_value))
        elif item[0] == 246 and item[1] == 246 and item[2] == 246:
            newData.append((250, 250, 250, alpha))
        elif item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((250, 250, 250, alpha))
        elif item[0] == 255 and item[1] == 253 and item[2] == 253:
            newData.append((250, 250, 250, alpha))
        else:
            newData.append(item)

    # Save new transparent plot in original location
    img.putdata(newData)
    img.save(image_filepath, "PNG",transparent=True)


def create_heatmaps(success_x,success_y,fail_x,fail_y,total_x,total_y,ep_str,orientation,state_rep,saving_dir,wrist_coords=None,finger_coords=None,title_str=""):
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
    heatmap_saving_dir = saving_dir + "heatmap_plots/"
    if not os.path.isdir(heatmap_saving_dir):
        os.mkdir(heatmap_saving_dir)

    freq_saving_dir = saving_dir + "freq_plots/"
    if not os.path.isdir(freq_saving_dir):
        os.mkdir(freq_saving_dir)

    if ep_str == "_":
        ep_str = ""
    elif ep_str != "":
        title_str = "\nEvaluated at Ep. " + ep_str
        ep_str = "_" + ep_str

    title_str += ", " + orientation.capitalize() + " Hand Orientation"

    if wrist_coords is None and finger_coords is None:
        if state_rep == "global" or state_rep == "local_to_global":
            # GLOBAL initial x,y coordinate positions of the hand
            palm_coords = [0.004856534244671696, 0.07105498718707715]
            f1_prox_coords = [0.05190709651841579, 0.05922107376362523]
            f2_prox_coords = [-0.047800791389975575, 0.05905594293043447]
            f3_prox_coords = [-0.047842290951250165, 0.05909756861129018]

            f1_dist_coords = [0.07998453867584633, 0.03696301651652619]
            f2_dist_coords = [-0.07601887636517492, 0.036798442851176685]
            f3_dist_coords = [-0.07606888985873166, 0.03684846417454405]
        elif state_rep == "local":
            # Initial (local) x,y coordinate positions of the hand
            palm_coords = [0, 0]
            f1_prox_coords = [-0.04704485301602504, 0.011828222834321257]
            f2_prox_coords = [0.052678133204167574, 0.011978223789074184]
            f3_prox_coords = [0.05267800707091165, 0.011978223789590757]

            f1_dist_coords = [-0.0751175118336241, 0.034081512699861705]
            f2_dist_coords = [0.08090042520228961, 0.03423151495416306]
            f3_dist_coords = [0.08090041735658912, 0.03423151495419517]
    else:
        palm_coords = wrist_coords[0:2]
        f1_prox_coords = finger_coords[0:3]
        f2_prox_coords = finger_coords[3:6]
        f3_prox_coords = finger_coords[6:9]

        f1_dist_coords = finger_coords[9:12]
        f2_dist_coords = finger_coords[12:15]
        f3_dist_coords = finger_coords[15:18]

    f1_line = [[f1_dist_coords[0],f1_prox_coords[0],palm_coords[0]], [f1_dist_coords[1],f1_prox_coords[1],palm_coords[1]]]
    f2_line = [[palm_coords[0],f2_prox_coords[0],f2_dist_coords[0]], [palm_coords[1],f2_prox_coords[1],f2_dist_coords[1]]]
    f3_line = [[palm_coords[0], f3_prox_coords[0], f3_dist_coords[0]], [palm_coords[1], f3_prox_coords[1], f3_dist_coords[1]]]
    hand_lines = {"finger1":f1_line, "finger2":f2_line, "finger3":f3_line}

    # Plot frequency heatmap
    freq_plot_title = "Grasp Trial Frequency per Initial Pose of Object\n" + title_str
    heatmap_freq(total_x,total_y,hand_lines,state_rep,freq_plot_title,'freq_heatmap'+ep_str+'.png',freq_saving_dir)

    actual_plot_title = "Initial Coordinate Position of the Object\n" + title_str
    heatmap_actual_coords(total_x,total_y, hand_lines, state_rep,freq_plot_title,'actual_heatmap'+ep_str+'.png',freq_saving_dir)

    # Plot failed (negative success rate) heatmap
    heatmap_plot_title = "Grasp Trial Success Rate per Initial Coordinate Position of the Object\n" + title_str
    heatmap_plot(success_x, success_y, fail_x, fail_y, total_x, total_y, hand_lines, state_rep, heatmap_plot_title, 'fail_heatmap'+ep_str+'.png',
                 heatmap_saving_dir, plot_success=False)

    # Plot successful heatmap
    heatmap_plot(success_x, success_y, fail_x, fail_y, total_x, total_y, hand_lines, state_rep,
                             heatmap_plot_title, 'success_heatmap'+ep_str+'.png', heatmap_saving_dir,plot_success=True)

    # Make heatmaps transparent to overlay over each other, then overlap them and save as a combine image
    success_filepath = heatmap_saving_dir + 'success_heatmap'+ep_str+'.png'
    fail_filepath = heatmap_saving_dir + 'fail_heatmap' + ep_str + '.png'
    combined_filepath = heatmap_saving_dir+'combined_heatmap'+ep_str+'.png'
    actual_heatmap_filepath = freq_saving_dir+'actual_heatmap'+ep_str+'.png'

    change_image_transparency(success_filepath)
    change_image_transparency(fail_filepath)
    change_image_transparency(actual_heatmap_filepath)

    overlap_images(success_filepath,fail_filepath, combined_filepath)

    # Once all heatmaps are generated, close the plots
    plt.close('all')


def overlap_images(img1_path, img2_path, save_filename):
    """ Overlap one image on top of another then change the background transparency to be opaque, then save.
    This is used to overlap success and failed image plots.
    img1_path: File path to the first image
    img2_path: File path to the second image
    """
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Overlap images
    img3 = Image.alpha_composite(img1, img2)

    img3.save(save_filename, "PNG", transparent=False)
    change_image_transparency(save_filename,alpha=1)


def get_heatmap_coord_data(data_dir,ep_str):
    """Get boxplot data from saved numpy arrays
    data_dir: Directory location of data file
    filename: Name of data file
    max_episodes: Total number of evaluation episodes
    saving_freq: Frequency in which the data files were saved
    """
    arr_dict = {}
    file_names = ["success_x","success_y","fail_x","fail_y","total_x","total_y"]
    for file in file_names:
        filename = data_dir+file+ep_str+".npy"
        my_file = Path(filename)
        if my_file.is_file():
            arr_dict[file] = np.load(filename)
        else:
            arr_dict[file] = np.array([])

    return arr_dict["success_x"], arr_dict["success_y"], arr_dict["fail_x"], arr_dict["fail_y"], arr_dict["total_x"], arr_dict["total_y"]


def generate_heatmaps(plot_type, orientation, data_dir, saving_dir, saving_freq=1000, max_episodes=20000, state_rep="local",title_str=""):
    """ Controls whether train or evaluation heatmap plots are generated
    plot_type: Type of plot (train or eval - eval will produce multiple plots)
    data_dir: Directory location of data file (coordinates)
    saving_dir: Directory to save the heatmap plots to
    saving_freq: Frequency at which the data was saved (used for getting filenames)
    max_episodes: Total number of episodes the data covers (to get up to the last data file)
    state_rep: Coordinate representation (local or global coordinate frame)
    """

    # If input data directory is not found, return back
    if not os.path.isdir(data_dir):
        print("data_dir not found! data_dir: ",data_dir)
        return

    # Create saving directory if it does not exist
    plot_save_path = Path(saving_dir)
    plot_save_path.mkdir(parents=True, exist_ok=True)

    # FOR EVAL
    if plot_type == "eval":
        for ep_num in np.linspace(start=0, stop=max_episodes, num=int(max_episodes / saving_freq)+1, dtype=int):
            print("EVAL EP NUM: ",ep_num)
            ep_str = str(ep_num)
            # Get coordinate data as numpy arrays
            success_x, success_y, fail_x, fail_y, total_x, total_y = get_heatmap_coord_data(data_dir, "_"+ep_str)

            # Plot coordinate data to frequency and success rate heatmaps
            create_heatmaps(success_x, success_y, fail_x, fail_y, total_x, total_y, ep_str, orientation, state_rep, saving_dir,title_str="")
    else:
        # FOR TRAIN
        ep_str = ""
        # Get coordinate data as numpy arrays
        success_x, success_y, fail_x, fail_y, total_x, total_y = get_heatmap_coord_data(data_dir, ep_str)
        total_x = np.append(success_x,fail_x)
        total_y = np.append(success_y,fail_y)

        # Plot coordinate data to frequency and success rate heatmaps
        create_heatmaps(success_x, success_y, fail_x, fail_y, total_x, total_y, ep_str, orientation, state_rep, saving_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, action='store', default="./")
    parser.add_argument("--saving_dir", type=str, action='store', default=None)
    parser.add_argument("--plot_type", type=str, action='store', default="train")
    parser.add_argument("--orientation", type=str, action='store', default="normal")
    parser.add_argument("--saving_freq", default=200, type=int) # Frequency at which the files were saved at (Determines the filename ex: success_x_1000.npy)
    parser.add_argument("--max_episodes", default=4000, type=int) # Maximum number of episodes to be plotted
    parser.add_argument("--state_rep", type=str, action='store', default="local") # Plots the positions of the fingers based on either the local or global coordinate frame
    args = parser.parse_args()

    data_dir = args.data_dir       # Directory to Get input data from
    if data_dir[-1] != "/":
        data_dir += "/"

    if args.saving_dir is None:
        saving_dir = data_dir
    else:
        saving_dir = args.saving_dir   # Directory to Save data to
    plot_type = args.plot_type     # Train or Eval plots
    orientation = args.orientation # Hand orientation
    saving_freq = args.saving_freq
    max_episodes = args.max_episodes
    state_rep = args.state_rep

    generate_heatmaps(plot_type, orientation, data_dir, saving_dir, saving_freq, max_episodes, state_rep)