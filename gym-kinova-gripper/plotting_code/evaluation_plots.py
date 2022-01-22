import matplotlib.pyplot as pyplt # Used for evaluation plotting
from heatmap_plot import create_heatmaps, overlap_images
import numpy as np
from pathlib import Path
import csv

def create_paths(dir_list):
    """ Create directories if they do not exist already, given path """
    for new_dir in dir_list:
        if new_dir is not None:
            new_path = Path(new_dir)
            new_path.mkdir(parents=True, exist_ok=True)

def reward_plot(eval_points, variation_input_policies, variation_input_name, policy_colors, start_episode, eval_freq, max_episode, shape_name="", orientation="", saving_dir=None):
    """ Plot the reward values from evaluation """
    reward_fig, axs = pyplt.subplots(1)
    reward_fig.set_size_inches(11, 8)
    reward_fig.suptitle("Success rate over 140 Grasp Trials per evaluation point (Every {} episodes)\nEvaluation with a {} shape and {} hand orientation".format(eval_freq,shape_name,orientation),fontsize=14)

    shape_marker_types = {"CubeM":'.', "CubeS":'d', "CubeB": 'D', "CylinderM": 'o', "Vase1M": 'v'}

    for policy_name, policy_rewards in variation_input_policies.items():
        if policy_colors.get(policy_name) is None:
            policy_colors[policy_name] = "green"

        for rewards_combo_dict in policy_rewards:
            reward_values = rewards_combo_dict["rewards"]
            rewards = [(reward/50)*100 for reward in reward_values]
            combo = rewards_combo_dict["orientation_shape"]
            if shape_marker_types.get(combo[1]) is None: # Default marker type is '.'
                shape_marker_types[combo[1]] = '.'

            if "naive" in policy_name:
                line_style = "dotted"
            elif "controller_a" in policy_name:
                line_style = '-.'
            else:
                line_style = '-'

            #label = policy_name + " " + combo[0] + ", " + combo[1]
            axs.plot(eval_points, rewards, label=policy_name, color=policy_colors[policy_name], marker=shape_marker_types[combo[1]], linestyle=line_style, linewidth=3)

    axs.set_xlabel("Evaluation Point (Episode)",fontsize=14)
    axs.set_ylabel("Grasp trial success rate %",fontsize=14)
    axs.legend(title="Evaluation Type", loc='best')
    tick_points = np.arange(start_episode, max_episode+1, step=max(1,eval_freq))
    axs.set_xticks(tick_points)
    axs.set_ylim(0, 100)
    axs.set_xlim(0, max_episode)

    pyplt.grid()
    if saving_dir is None:
        reward_fig.show()
    else:
        reward_filepath = saving_dir + "/"+variation_input_name+"_"+shape_name+"_"+orientation

        print("*** Writing reward plot and values to: ",reward_filepath)
        dict_file = open(reward_filepath + "reward_plot_policy_rewards.csv", "w", newline='')
        keys = variation_input_policies.keys()
        dict_writer = csv.DictWriter(dict_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows([variation_input_policies])
        dict_file.close()

        reward_fig.savefig(reward_filepath+"_Reward_Evaluation_Plot.png")
        pyplt.close()


def generate_heatmaps_by_orientation_frame(variation_type, shape, orientation, coord_frames, ep_num, all_hand_object_coords, variation_heatmap_path):
    """ Generates heatmaps based on the region of the object's initial x-coordinate position within the hand.
    This function will take in all of the hand-object coordinates generated from evaluating the policy, sorts them
    by region, and generates new heatmap plots for each of the regions.
    """
    ep_str = ""
    if orientation == "random":
        orientations_list = ["normal", "rotated", "top"]
    else:
        orientations_list = [orientation]

    for orientation in orientations_list:
        heatmap_orient_dir = variation_heatmap_path + orientation + "/"
        heatmap_shape_dir = variation_heatmap_path + orientation + "/" + shape + "/"

        hand_object_coords_dicts = [d for d in all_hand_object_coords if d["orientation"] == orientation and d["shape"] == shape]

        ## PLOT ALL COORDINATES BY GLOBAL/LOCAL FRAME
        # Potential cooridnate frames include "local", "global", "local_to_global"
        for frame in coord_frames:
            print("Generating heatmaps within the {} frame!!\nOrientation: {}\nShape: {}\n".format(frame, orientation,shape))
            create_paths([variation_heatmap_path, heatmap_orient_dir, heatmap_shape_dir, heatmap_shape_dir + frame + "/"])
            success_coords = [d[frame + "_obj_coords"] for d in hand_object_coords_dicts if d["success"] is True]
            fail_coords = [d[frame + "_obj_coords"] for d in hand_object_coords_dicts if d["success"] is False]

            wrist_coords = None
            finger_coords = None

            success_x = [coords[0] for coords in success_coords]
            success_y = [coords[1] for coords in success_coords]
            fail_x = [coords[0] for coords in fail_coords]
            fail_y = [coords[1] for coords in fail_coords]
            total_x = success_x + fail_x
            total_y = success_y + fail_y

            ep_str = create_heatmaps(success_x, success_y, fail_x, fail_y, total_x, total_y, shape, orientation, ep_num=ep_num,
                            wrist_coords=wrist_coords, finger_coords=finger_coords, state_rep=frame,
                            saving_dir=heatmap_shape_dir + frame + "/",
                            title_str="Input variation: "+variation_type["variation_name"] + ", " + frame.capitalize() + " Coord. Frame")

        #local_actual_filepath = heatmap_shape_dir + "local/" + "freq_plots/" + 'actual_heatmap' + ep_str + '.png'
        #global_actual_filepath = heatmap_shape_dir + "global/" + "freq_plots/" + 'actual_heatmap' + ep_str + '.png'
        #local_to_global_actual_filepath = heatmap_shape_dir + "local_to_global/" + "freq_plots/" + 'actual_heatmap' + ep_str + '.png'

        #combined_actual_filepath = heatmap_shape_dir + 'local_vs_global_combined_actual_heatmap' + ep_str + '.png'
        #overlap_images(local_actual_filepath, global_actual_filepath, combined_actual_filepath)

        """
        local_to_global_combined_filepath = heatmap_orient_dir + 'local_to_global_vs_global_combined_actual_heatmap.png'
        overlap_images(local_to_global_actual_filepath, global_actual_filepath, local_to_global_combined_filepath)
        """