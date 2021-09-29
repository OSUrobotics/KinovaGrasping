import gym
import os,sys
import numpy as np
import copy
import csv
import DDPGfD
import matplotlib.pyplot as plt
from main_DDPGfD import eval_policy, setup_directories
# Import plotting code from other directory
plot_path = os.getcwd() + "/plotting_code"
sys.path.insert(1, plot_path)
from heatmap_plot import heatmap_freq


def get_coord_file_indexes(env,shape_name,hand_orientation,with_noise,orient_idx=None):
    """
        Get a list of file indexes based on the number of lines in the object-hand pose coordinate file
        (per shape/size/hand orientation)
    """
    env.set_orientation(hand_orientation)
    if orient_idx is None:
        _, _, _, _, _, _, _, coords_file = env.determine_obj_hand_coords(random_shape=shape_name, mode="shape", orient_idx=0, with_noise=with_noise)
        num_lines = 0
        with open(coords_file) as f:
            for line in f:
                num_lines = num_lines + 1
        indexes = range(num_lines)
    else:
        indexes = [orient_idx]

    coords_file_directory = os.path.dirname(coords_file)

    return indexes, coords_file, coords_file_directory


def determine_obj_hand_pose_difficulty(curr_orient_idx,shape_name,hand_orientation,with_noise,max_num_timesteps,policy,all_saving_dirs):
    """
        Determine the difficulty of an object-hand pose based on the constant-speed controller, variable-speed controller,
        and policy's ability to successfully grasp and lift the obejct.
    """
    seed = 2
    velocities = {"constant_velocity": 2, "min_velocity": 0, "max_velocity": 3, "finger_lift_velocity": 1,
                  "wrist_lift_velocity": 1}
    controller_options = ["naive","position-dependent","policy"]

    # Store object-hand pose info, incl. if controllers are successful given this pose and subsequent difficulty label
    coord_info = {"obj_coord":[],"hov":[],"shape":shape_name,"hand_orientation":hand_orientation,"naive_success":None,"position-dependent_success":None,"policy_success":None,"difficulty":None}

    # Attempt a grasp trial with each controller
    for controller_type in controller_options:
        eval_ret = eval_policy(policy, env_name, seed,
                                      requested_shapes=[shape_name],
                                      requested_orientation=hand_orientation,
                                      eval_episodes=1,
                                      render_imgs=False,
                                      all_saving_dirs=all_saving_dirs,
                                      velocities=velocities,
                                      output_dir=all_saving_dirs["output_dir"],
                                      with_noise=with_noise,
                                      orient_idx=curr_orient_idx,
                                      max_num_timesteps=max_num_timesteps,
                                      controller_type=controller_type,
                                      mode="shape",
                                      state_idx_arr=np.arange(82))

        hand_object_coords = eval_ret["all_hand_object_coords"][0]
        coord_info["obj_coord"] = hand_object_coords["global_obj_coords"]
        coord_info["hov"] = hand_object_coords["hand_orient_variation"]
        coord_info[controller_type+"_success"] = hand_object_coords["success"]

    # Determine the difficulty of the pose based on the amount off success
    success_sum = coord_info["naive_success"] + coord_info["position-dependent_success"] + coord_info["policy_success"]
    if success_sum is 3:
        coord_info["difficulty"] = "easy"
    elif 1 <= success_sum <= 2:
        coord_info["difficulty"] = "med"
    elif success_sum is 0:
        coord_info["difficulty"] = "hard"

    return coord_info


def loop_through_coord_file(indexes, shape_name, hand_orientation, with_noise, max_num_timesteps, policy, all_saving_dirs):
    """
        Loop through each object-hand pose coordinates, determine its difficulty, and store that information in a dictionary
    """
    labelled_obj_hand_coords = [] # List of coordinates and their difficulty

    for curr_orient_idx in indexes:
        print("Coord File Idx: ", curr_orient_idx)
        coord_info = determine_obj_hand_pose_difficulty(curr_orient_idx, shape_name, hand_orientation, with_noise, max_num_timesteps, policy, all_saving_dirs)
        labelled_obj_hand_coords.append(coord_info)

    return labelled_obj_hand_coords

def plot_coords_by_difficulty(labelled_obj_hand_coords,saving_dir=None):
    """
    Sort the coordinates by difficulty and plot their frequency per difficulty (easy,med,hard)
    """
    easy_coords = [d["obj_coord"] for d in labelled_obj_hand_coords if d["difficulty"] == "easy"]
    med_coords = [d["obj_coord"] for d in labelled_obj_hand_coords if d["difficulty"] == "med"]
    hard_coords = [d["obj_coord"] for d in labelled_obj_hand_coords if d["difficulty"] == "hard"]
    all_coords_x = [d["obj_coord"][0] for d in labelled_obj_hand_coords]
    all_coords_y = [d["obj_coord"][1] for d in labelled_obj_hand_coords]
    coords_by_difficulty = {"Easy":{"coords":easy_coords,"color_map":plt.cm.Greens},"Medium":{"coords":med_coords,"color_map":plt.cm.Blues},"Hard":{"coords":hard_coords,"color_map":plt.cm.Reds}}

    # Plot all coordinates frequency
    freq_plot_title = "Frequency of all object-hand pose coordinates"
    heatmap_freq(all_coords_x, all_coords_y, hand_lines=None, state_rep="global", plot_title=freq_plot_title, fig_filename=None, saving_dir=None)

    for difficulty,coords in coords_by_difficulty.items():
        x_coords = [c[0] for c in coords["coords"]]
        y_coords = [c[1] for c in coords["coords"]]

        # Plot frequency heatmap
        freq_plot_title = "Frequency of object-hand pose coordinates with an " + difficulty + " grasp trial difficulty"
        heatmap_freq(x_coords, y_coords, hand_lines=None, state_rep="global", plot_title=freq_plot_title, fig_filename=None,saving_dir=None,color_map=coords['color_map'])

    # Plot each of the controllers by difficulty
    difficulty_bar_plot(coords_by_difficulty)

    # Success/failure heatmap per controller


def difficulty_bar_plot(coords_by_difficulty):
    labels = ['normal']
    difficulty_by_controller = {"easy":0,"medium":0,"hard":0}
    difficulty_by_controller["easy"] = len(coords_by_difficulty["Easy"]["coords"])
    difficulty_by_controller["med"] = len(coords_by_difficulty["Medium"]["coords"])
    difficulty_by_controller["hard"] = len(coords_by_difficulty["Hard"]["coords"])

    #for difficulty in ["easy","med","hard"]:
    #    difficulty_by_controller[difficulty] = [sum([d["naive_success"] for d in labelled_obj_hand_coords if d["difficulty"] == difficulty]),
    #     sum([d["position-dependent_success"] for d in labelled_obj_hand_coords if d["difficulty"] == difficulty]),
    #     sum([d["policy_success"] for d in labelled_obj_hand_coords if d["difficulty"] == difficulty])]

    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    easy_rect = ax.bar(x + 0, difficulty_by_controller["easy"], width=0.25, label='Easy')
    med_rect = ax.bar(x + 0.25, difficulty_by_controller["med"], width=0.25, label='Medium')
    hard_rect = ax.bar(x + 0.50, difficulty_by_controller["hard"], width=0.25, label='Hard')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Hand Orientation')
    ax.set_title('Object-hand pose success frequency by difficulty per controller')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    rects = ax.patches
    labels = [difficulty_by_controller[difficulty] for difficulty in ["easy","med","hard"]]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.25, label, ha="center", va="bottom")

    fig.tight_layout()

    plt.show()
    
    
def write_obj_hand_pose_dict_list(saving_dir,labelled_obj_hand_coords):
    """
    Write the object-hand pose dictionary to a csv file
    """
    dict_file = open(saving_dir + "/labelled_obj_hand_coords.csv", "w", newline='')
    keys = labelled_obj_hand_coords[0].keys()
    dict_writer = csv.DictWriter(dict_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(labelled_obj_hand_coords)
    dict_file.close()


def read_obj_hand_pose_dict_list(saving_dir):
    """
    Read the object-hand pose dictionary from a csv file
    """
    labelled_obj_hand_coords = []
    with open(saving_dir + "/labelled_obj_hand_coords.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labelled_obj_hand_coords.append(row)
    return labelled_obj_hand_coords


if __name__ == "__main__":
    # TODO: Make into command-line arguments
    coords_option = "label"

    with_noise = True
    policy_filepath = "./experiments/pre-train/Pre-train_Baseline_FULL_DIM/policy/pre-train_DDPGfD_kinovaGrip"
    all_shapes = ["CubeM"] #["CubeS", "CubeM", "CubeB", "CylinderM", "Vase1M"]
    all_orientations = ["normal"] #["normal", "top", "rotated"]

    # Initialize the environment
    env_name = 'gym_kinova_gripper:kinovagripper-v0'
    env = gym.make(env_name)  # Initialize environment
    max_num_timesteps = copy.deepcopy(env._max_episode_steps)

    if coords_option == "label":
        # Go through all object-hand pose coordinates and label them by difficulty (easy, med, hard)
        for hand_orientation in all_orientations:
            for shape_name in all_shapes:
                # Initialize the policy
                policy = DDPGfD.DDPGfD()
                policy.sampling_decay_rate = 0
                policy.load(policy_filepath)

                # Get the object-hand pose coordinate file indexes
                indexes, coords_file, coords_file_directory = get_coord_file_indexes(env,shape_name,hand_orientation,with_noise)

                # Setup the output directories
                coords_saving_dir = coords_file_directory + "/" + shape_name + "/"
                all_saving_dirs = setup_directories(env, saving_dir=coords_saving_dir, expert_replay_file_path=None, agent_replay_file_path=None, pretrain_model_save_path=None, create_dirs=True,mode="eval")

                # Go through each of the object-hand coordinates
                labelled_obj_hand_coords = loop_through_coord_file(indexes, shape_name, hand_orientation, with_noise, max_num_timesteps, policy, all_saving_dirs)

                # Write coordinate difficulty info to a file
                write_obj_hand_pose_dict_list(all_saving_dirs["output_dir"],labelled_obj_hand_coords)

    elif coords_option == "plot":
        # Plot each object-hand pose coordinates by difficulty per object size/shape and hand orientation
        for hand_orientation in all_orientations:
            for shape_name in all_shapes:
                # Get the object-hand pose coordinate file indexes
                indexes, coords_file, coords_file_directory = get_coord_file_indexes(env,shape_name,hand_orientation,with_noise)

                output_saving_dir = coords_saving_dir = coords_file_directory + "/" + shape_name + "/" + "/output/"

                # Read in pre-labelled coordinate dictionaries from a file
                labelled_obj_hand_coords = read_obj_hand_pose_dict_list(output_saving_dir)

                # Plot coordinates based on difficulty (bar chart, heatmap)
                #plot_coords_by_difficulty(labelled_obj_hand_coords)