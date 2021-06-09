import matplotlib.pyplot as pyplt # Used for evaluation plotting
from heatmap_plot import create_heatmaps, overlap_images
import numpy as np
from pathlib import Path

def create_paths(dir_list):
    """ Create directories if they do not exist already, given path """
    for new_dir in dir_list:
        if new_dir is not None:
            new_path = Path(new_dir)
            new_path.mkdir(parents=True, exist_ok=True)

def reward_plot(eval_points, variation_input_policies, variation_input_name, policy_colors, eval_freq, max_episode, saving_dir):
    """ Plot the reward values from evaluation """
    reward_fig, axs = pyplt.subplots(1)
    reward_fig.suptitle("Variation Input: {}\nAvg. Reward from 500 Grasp Trials per evaluation point (Every {} episodes)\nEvaluation over each policy variation type".format(variation_input_name,eval_freq))

    for policy_name, rewards in variation_input_policies.items():
        if policy_colors.get(policy_name) is None:
            policy_colors[policy_name] = "green"
        axs.plot(eval_points, rewards, label=policy_name, color=policy_colors[policy_name])

    axs.set_xlabel("Evaluation Point (Episode)")
    axs.set_ylabel("Reward")
    axs.legend(title="Policy Type", loc='best')
    tick_points = np.arange(0, max_episode+1, step=max(1,eval_freq))
    axs.set_xticks(tick_points)
    axs.set_ylim(0, 55)
    axs.set_xlim(0, max_episode)

    pyplt.grid()
    reward_fig.savefig(saving_dir + "/"+variation_input_name+"_Reward_Evaluation_Plot.png")
    pyplt.close()


def generate_heatmaps_by_orientation_frame(variation_type,all_hand_object_coords, variation_heatmap_path):
    """ Generates heatmaps based on the region of the object's initial x-coordinate position within the hand.
    This function will take in all of the hand-object coordinates generated from evaluating the policy, sorts them
    by region, and generates new heatmap plots for each of the regions.
    """
    requested_orientation = variation_type["requested_orientation"]
    if requested_orientation == "random":
        orientations_list = ["normal", "rotated", "top"]
    else:
        orientations_list = ["normal"]

    for orientation in orientations_list:
        hand_object_coords_dicts = [d for d in all_hand_object_coords if d["orientation"] == orientation]
        heatmap_orient_dir = variation_heatmap_path + orientation + "/"

        ## PLOT ALL COORDINATES BY GLOBAL/LOCAL FRAME
        coord_frames = ["local", "global"] #, "local_to_global"]
        for frame in coord_frames:
            create_paths([variation_heatmap_path, heatmap_orient_dir, heatmap_orient_dir + frame + "/"])
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

            create_heatmaps(success_x, success_y, fail_x, fail_y, total_x, total_y, "", orientation,
                            wrist_coords=wrist_coords, finger_coords=finger_coords, state_rep=frame,
                            saving_dir=heatmap_orient_dir + frame + "/",
                            title_str="Input variation: "+variation_type["variation_name"] + ", " + frame.capitalize() + " Coord. Frame")


        local_actual_filepath = heatmap_orient_dir + "local/" + "freq_plots/" + 'actual_heatmap.png'
        global_actual_filepath = heatmap_orient_dir + "global/" + "freq_plots/" + 'actual_heatmap.png'
        local_to_global_actual_filepath = heatmap_orient_dir + "local_to_global/" + "freq_plots/" + 'actual_heatmap.png'

        combined_actual_filepath = heatmap_orient_dir + 'local_vs_global_combined_actual_heatmap.png'
        overlap_images(local_actual_filepath, global_actual_filepath, combined_actual_filepath)

        """
        local_to_global_combined_filepath = heatmap_orient_dir + 'local_to_global_vs_global_combined_actual_heatmap.png'
        overlap_images(local_to_global_actual_filepath, global_actual_filepath, local_to_global_combined_filepath)
        """