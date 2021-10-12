import gym
import os,sys
import numpy as np
import copy
import csv
import json
import DDPGfD
import matplotlib.pyplot as plt
from main_DDPGfD import eval_policy, setup_directories
# Import plotting code from other directory
plot_path = os.getcwd() + "/plotting_code"
sys.path.insert(1, plot_path)
from heatmap_plot import heatmap_freq
from evaluation_plots import reward_plot


def get_coord_file_indexes(env,shape_name,hand_orientation,with_noise,coords_type="shape",orient_idx=None):
    """
        Get a list of file indexes based on the number of lines in the object-hand pose coordinate file
        (per shape/size/hand orientation)
    """
    env.set_orientation(hand_orientation)
    if orient_idx is None:
        _, _, _, _, _, _, _, coords_file = env.determine_obj_hand_coords(random_shape=shape_name, mode=coords_type, orient_idx=0, with_noise=with_noise)
        num_lines = 0
        with open(coords_file) as f:
            for line in f:
                num_lines = num_lines + 1
        indexes = range(num_lines)
    else:
        indexes = [orient_idx]

    coords_file_directory = os.path.dirname(coords_file)

    return indexes, coords_file, coords_file_directory

def get_difficulty_label(coord_info):
    """
    Determine the difficulty of the pose based on the amount off success
    """
    if coord_info["naive_success"] == 'True' or coord_info["naive_success"] == 'TRUE' or coord_info["naive_success"] is True:
        naive_success = 1
    else:
        naive_success = 0

    if coord_info["position-dependent_success"] == 'True' or coord_info["position-dependent_success"] == 'TRUE' or coord_info["position-dependent_success"] is True:
        position_dependent_success = 1
    else:
        position_dependent_success = 0

    success_sum = naive_success + position_dependent_success
    if success_sum is 2:
        return "easy"
    elif success_sum is 1:
        return "med"
    elif success_sum is 0:
        return "hard"

def conduct_grasp_trial(coord_info,curr_orient_idx,shape_name,hand_orientation,with_noise,coords_type,max_num_timesteps,all_saving_dirs):
    """
    Conduct a grasp trial to determine the controllers' performance with a given object-hand pose coordinate pair.
    """
    seed = 2
    velocities = {"constant_velocity": 2, "min_velocity": 0, "max_velocity": 3, "finger_lift_velocity": 1,
                  "wrist_lift_velocity": 1}
    controller_options = ["naive","position-dependent"]

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
                                      mode=coords_type,
                                      state_idx_arr=np.arange(82))

        hand_object_coords = eval_ret["all_hand_object_coords"][0]
        coord_info["obj_coords"] = hand_object_coords["obj_coords"]
        coord_info["local_obj_coords"] = hand_object_coords["local_obj_coords"]
        coord_info["hov"] = hand_object_coords["hov"]
        coord_info[controller_type+"_success"] = hand_object_coords["success"]

    return coord_info


def determine_obj_hand_pose_difficulty(curr_orient_idx,shape_name,hand_orientation,with_noise,max_num_timesteps,all_saving_dirs,pre_labelled_obj_hand_coords,coords_type="shape"):
    """
        Determine the difficulty of an object-hand pose based on the constant-speed controller and variable-speed controller's
        ability to successfully grasp and lift the obejct.
    """
    # Store object-hand pose info, incl. if controllers are successful given this pose and subsequent difficulty label
    coord_info = {"obj_coords":[],"local_obj_coords":[],"hov":[],"shape":shape_name,"hand_orientation":hand_orientation,"naive_success":None,"position-dependent_success":None,"policy_success":None,"difficulty":None}

    if pre_labelled_obj_hand_coords is not None:
        coord_info = pre_labelled_obj_hand_coords[curr_orient_idx]
    else:
        coord_info = conduct_grasp_trial(coord_info,curr_orient_idx,shape_name,hand_orientation,with_noise,coords_type,max_num_timesteps,all_saving_dirs)

    coord_info.pop("policy_success", None)
    coord_info["difficulty"] = get_difficulty_label(coord_info)

    return coord_info


def loop_through_coord_file(indexes, shape_name, hand_orientation, with_noise, max_num_timesteps, policy, all_saving_dirs, pre_labelled_obj_hand_coords, coords_type):
    """
        Loop through each object-hand pose coordinates, determine its difficulty, and store that information in a dictionary
    """
    labelled_obj_hand_coords = [] # List of coordinates and their difficulty

    for curr_orient_idx in indexes:
        print("Coord File Idx: ", curr_orient_idx)
        coord_info = determine_obj_hand_pose_difficulty(curr_orient_idx, shape_name, hand_orientation, with_noise, max_num_timesteps, all_saving_dirs, pre_labelled_obj_hand_coords, coords_type)
        labelled_obj_hand_coords.append(coord_info)

    return labelled_obj_hand_coords

def plot_coords_by_difficulty(labelled_obj_hand_coords,eval_point_str="",saving_dir=None):
    """
    Sort the coordinates by difficulty and plot their frequency per difficulty (easy,med,hard)
    """
    easy_coords = [d["local_obj_coords"] for d in labelled_obj_hand_coords if d["difficulty"] == "easy"]
    med_coords = [d["local_obj_coords"] for d in labelled_obj_hand_coords if d["difficulty"] == "med"]
    hard_coords = [d["local_obj_coords"] for d in labelled_obj_hand_coords if d["difficulty"] == "hard"]
    all_coords_x = [d["local_obj_coords"][0] for d in labelled_obj_hand_coords]
    all_coords_y = [d["local_obj_coords"][1] for d in labelled_obj_hand_coords]
    coords_by_difficulty = {"Easy":{"coords":easy_coords,"color_map":plt.cm.Greens},"Medium":{"coords":med_coords,"color_map":plt.cm.Blues},"Hard":{"coords":hard_coords,"color_map":plt.cm.Reds}}

    # Plot all coordinates frequency
    freq_plot_title = "Frequency of all object-hand pose coordinates"
    heatmap_freq(all_coords_x, all_coords_y, hand_lines=None, state_rep="global", plot_title=freq_plot_title, fig_filename=eval_point_str + "all_coord_frequency.png", saving_dir=saving_dir, y_min=0)

    for difficulty,coords in coords_by_difficulty.items():
        x_coords = [c[0] for c in coords["coords"]]
        y_coords = [c[1] for c in coords["coords"]]

        # Plot frequency heatmap
        freq_plot_title = "Frequency of object-hand pose coordinates with an " + difficulty + " grasp trial difficulty"
        heatmap_freq(x_coords, y_coords, hand_lines=None, state_rep="global", plot_title=freq_plot_title, fig_filename=eval_point_str + difficulty +"_coord_frequency.png", saving_dir=saving_dir,color_map=coords['color_map'],y_min=0)

    # Plot each of the controllers by difficulty
    difficulty_bar_plot(coords_by_difficulty,eval_point_str + "difficulty_bar_plot.png",saving_dir)

    # Success/failure heatmap per controller


def difficulty_bar_plot(coords_by_difficulty,fig_filename,saving_dir):
    labels = ['normal']
    difficulty_by_controller = {"easy":0,"medium":0,"hard":0}
    difficulty_by_controller["easy"] = len(coords_by_difficulty["Easy"]["coords"])
    difficulty_by_controller["med"] = len(coords_by_difficulty["Medium"]["coords"])
    difficulty_by_controller["hard"] = len(coords_by_difficulty["Hard"]["coords"])

    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    easy_rect = ax.bar(x + 0, difficulty_by_controller["easy"], width=0.25, color='green', label='Easy')
    med_rect = ax.bar(x + 0.25, difficulty_by_controller["med"], width=0.25, color='blue', label='Medium')
    hard_rect = ax.bar(x + 0.50, difficulty_by_controller["hard"], width=0.25, color='red', label='Hard')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Hand Orientation')
    ax.set_title('Object-hand pose dataset frequency by difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    rects = ax.patches
    labels = [difficulty_by_controller[difficulty] for difficulty in ["easy","med","hard"]]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.25, label, ha="center", va="bottom")

    fig.tight_layout()

    if saving_dir is None:
        plt.show()
    else:
        plt.savefig(saving_dir+fig_filename)
        plt.close(fig)
    
    
def write_obj_hand_pose_dict_list(saving_dir,filename,labelled_obj_hand_coords):
    """
    Write the object-hand pose dictionary to a csv file
    """
    dict_file = open(saving_dir + "/" + filename, "w", newline='')
    keys = labelled_obj_hand_coords[0].keys()
    dict_writer = csv.DictWriter(dict_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(labelled_obj_hand_coords)
    dict_file.close()


def read_obj_hand_pose_dict_list(saving_dir,num_coords=None,filename="labelled_obj_hand_coords.csv"):
    """
    Read the object-hand pose dictionary from a csv file
    """
    labelled_obj_hand_coords = []
    idx = 0
    with open(saving_dir + filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if num_coords is not None and idx >= num_coords:
                break
            obj_coords = row["obj_coords"].strip('][').split(', ')
            row["obj_coords"] = [float(coord) for coord in obj_coords]
            local_obj_coords = row["local_obj_coords"].strip('][').split(', ')
            row["local_obj_coords"] = [float(coord) for coord in local_obj_coords]

            hov = row["hov"].strip('][')
            hov = hov.split(' ')
            hov = [h.strip(',') for h in hov if h != ' ' and h != '']
            row["hov"] = [float(coord) for coord in hov]

            if len(row["difficulty"]) > 5:
                diff = row["difficulty"].replace("'", "\"")
                d = json.loads(diff)
                row["difficulty"] = d["difficulty"]
                row["naive_success"] = d["naive_success"]
                row["position-dependent_success"] = d["position-dependent_success"]
                row["policy_success"] = d["policy_success"]

            idx += 1
            labelled_obj_hand_coords.append(row)

    return labelled_obj_hand_coords


def get_difficulty_success_rate(labelled_obj_hand_coords):
    """
    Get the success rate per coordinate difficulty
    """
    difficulty_success_rate = {"easy":0,"med":0,"hard":0}

    num_easy = sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["difficulty"] == "easy"])
    num_med = sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["difficulty"] == "med"])
    num_hard = sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["difficulty"] == "hard"])

    if num_easy > 0:
        difficulty_success_rate["easy"] = (sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["difficulty"] == "easy" and obj_coord["success"] == 'True']) / num_easy)*50
    else:
        difficulty_success_rate["easy"] = 0
    if num_med > 0:
        difficulty_success_rate["med"] = (sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["difficulty"] == "med" and obj_coord["success"] == 'True']) / num_med)*50
    else:
        difficulty_success_rate["med"] = 0
    if num_hard > 0:
        difficulty_success_rate["hard"] = (sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["difficulty"] == "hard" and obj_coord["success"] == 'True']) / num_hard)*50
    else:
        difficulty_success_rate["hard"] = 0
    difficulty_success_rate["all_coords"] = (sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["success"] == 'True']) / len(labelled_obj_hand_coords))*50
    difficulty_success_rate["naive"] = (sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["naive_success"] == 'True']) / len(labelled_obj_hand_coords)) * 50
    difficulty_success_rate["position-dependent"] = (sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["position-dependent_success"] == 'True']) / len(labelled_obj_hand_coords)) * 50
    difficulty_success_rate["pre-trained_policy"] = (sum([1 for obj_coord in labelled_obj_hand_coords if obj_coord["policy_success"] == 'True']) / len(labelled_obj_hand_coords)) * 50

    return difficulty_success_rate


def plot_success_rate_by_difficulty(all_policy_coords, eval_points, start_episode, eval_freq, max_episode, orientation, shape_name, variation_input_name, saving_dir):
    line_types = ["all_coords","easy","med","hard","naive","position-dependent","pre-trained_policy"]
    policy_colors = {"all_coords": "black", "easy": "green","med": "blue", "hard": "red", "naive": "grey", "position-dependent": "grey", "pre-trained_policy": "grey"}
    variation_input_policies = {}

    for type in line_types:
        variation_input_policies[type] = [{"rewards": [policy_coords["difficulty"][type] for policy_coords in all_policy_coords.values()],"orientation_shape":[orientation,shape_name]}]

    reward_plot(eval_points, variation_input_policies, variation_input_name, policy_colors, start_episode, eval_freq, max_episode, shape_name, orientation, saving_dir)


def get_variation_input(variation_input_name):
    """
    Return a dictionary of info about the variation input given the name.
    """
    # Evaluation input variations
    Baseline = {"variation_name": "Baseline", "requested_shapes": ["CubeM"], "requested_orientation": "normal",
                "with_orientation_noise": False}
    Baseline_HOV = {"variation_name": "Baseline_HOV", "requested_shapes": ["CubeM"], "requested_orientation": "normal",
                    "with_orientation_noise": True}
    Sizes_HOV = {"variation_name": "Sizes_HOV", "requested_shapes": ["CubeS", "CubeM", "CubeB"],
                 "requested_orientation": "normal", "with_orientation_noise": True}
    Shapes_HOV = {"variation_name": "Shapes_HOV", "requested_shapes": ["CubeM", "CylinderM", "Vase1M"],
                  "requested_orientation": "normal", "with_orientation_noise": True}
    Orientations_HOV = {"variation_name": "Orientations_HOV", "requested_shapes": ["CubeM"],
                        "requested_orientation": "random", "with_orientation_noise": True}

    # Contains all input variation types
    variations_dict = {"Baseline": Baseline, "Baseline_HOV": Baseline_HOV, "Sizes_HOV": Sizes_HOV,
                       "Shapes_HOV": Shapes_HOV, "Orientations_HOV": Orientations_HOV}

    return variations_dict[variation_input_name]


if __name__ == "__main__":
    # TODO: Make into command-line arguments
    coords_option = "plot_coords"
    start_episode, eval_freq, max_episode = 0, 400, 10000
    with_noise = False
    variation_input_name = "Baseline"
    policy_filepath = "./experiments/pre-train/Pre-train_Baseline_FULL_DIM/policy/pre-train_DDPGfD_kinovaGrip"
    #policy_filepath = "./experiments/eval/Eval_Pre-Trained_Policy_400/" + variation_input_name
    #policy_filepath = "./experiments/eval/Eval_Pre-Trained_Policy_Diff/" + variation_input_name
    variation_input_dict = get_variation_input(variation_input_name)

    all_shapes = ["CubeM"] #["CubeS", "CubeM", "CubeB", "CylinderM", "Vase1M"]
    all_orientations = ["normal"] #["normal", "top", "rotated"]
    coords_type = "train"

    if start_episode is None or eval_freq is None or max_episode is None:
        plot_policies_over_time = False
    else:
        plot_policies_over_time = True

    if with_noise is True:
        noise_str = "with_noise"
    else:
        noise_str = "no_noise"

    # Get labelled coordinates from the file to re-label them by difficulty
    get_labelled_coords_from_file = False

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
                indexes, coords_file, coords_file_directory = get_coord_file_indexes(env,shape_name,hand_orientation,with_noise,coords_type)

                # Setup the output directories
                coords_saving_dir = coords_file_directory + "/" + shape_name + "/labelled_coords/"
                all_saving_dirs = setup_directories(env, saving_dir=coords_saving_dir, expert_replay_file_path=None, agent_replay_file_path=None, pretrain_model_save_path=None, create_dirs=True,mode="eval")

                if get_labelled_coords_from_file is True:
                    pre_labelled_obj_hand_coords = read_obj_hand_pose_dict_list(coords_saving_dir)
                    filename = "NEW_labelled_obj_hand_coords.csv"
                else:
                    pre_labelled_obj_hand_coords = None
                    filename = "labelled_obj_hand_coords.csv"

                # Go through each of the object-hand coordinates
                labelled_obj_hand_coords = loop_through_coord_file(indexes, shape_name, hand_orientation, with_noise, max_num_timesteps, policy, all_saving_dirs, pre_labelled_obj_hand_coords, coords_type)

                # Write coordinate difficulty info to a file
                write_obj_hand_pose_dict_list(all_saving_dirs["saving_dir"],filename,labelled_obj_hand_coords)

    elif coords_option == "plot_coords":
        # Plot each object-hand pose coordinates by difficulty per object size/shape and hand orientation
        for hand_orientation in all_orientations:
            for shape_name in all_shapes:
                # Get the object-hand pose coordinate file indexes

                coords_file_directory = "./gym_kinova_gripper/envs/kinova_description/obj_hand_coords/"+noise_str+"/"+coords_type+"_coords/"+hand_orientation+"/"

                output_saving_dir = coords_saving_dir = coords_file_directory + "/" + shape_name + "/labelled_coords/"

                # Read in pre-labelled coordinate dictionaries from a file
                labelled_obj_hand_coords = read_obj_hand_pose_dict_list(output_saving_dir)

                # Plot coordinates based on difficulty (bar chart, heatmap)
                plot_coords_by_difficulty(labelled_obj_hand_coords,saving_dir=output_saving_dir)

    elif coords_option == "plot_policy_coords":
        all_policy_coords = {}
        if plot_policies_over_time is True:
            if eval_freq == 0:
                policy_eval_points = np.array([0])
            else:
                num_policies = int((max_episode - start_episode) / eval_freq) + 1
                policy_eval_points = np.linspace(start=start_episode, stop=max_episode, num=num_policies, dtype=int)

            for eval_point in policy_eval_points:
                eval_point_filepath = policy_filepath + "/policy_" + str(eval_point) + "/output/"

                # Read in pre-labelled coordinate dictionaries from a file
                labelled_obj_hand_coords = read_obj_hand_pose_dict_list(eval_point_filepath,num_coords=100,filename="all_hand_object_coords.csv")

                difficulty_success_rate = get_difficulty_success_rate(labelled_obj_hand_coords)

                # Collect coordinate data from each evaluation point
                all_policy_coords[str(eval_point)] = {"coords": labelled_obj_hand_coords, "difficulty": difficulty_success_rate}

                # Plot coordinates based on difficulty (bar chart, heatmap)
                plot_coords_by_difficulty(labelled_obj_hand_coords,eval_point_str=str(eval_point)+"_",saving_dir=policy_filepath+"/policy_"+str(eval_point)+"/")

            # With all coordinate data over time collected, plot each evaluation point by difficulty
            for orientation in [variation_input_dict["requested_orientation"]]:
                for shape_name in variation_input_dict["requested_shapes"]:
                    eval_policy_coords = {}
                    for eval_point, eval_coords_dict in all_policy_coords.items():
                        shape_orient_policy_coords = []
                        for coords_dict in eval_coords_dict["coords"]:
                            if coords_dict["shape"] == shape_name and coords_dict["orientation"] == orientation:
                                shape_orient_policy_coords.append(coords_dict)
                        eval_difficulty_success_rate = get_difficulty_success_rate(shape_orient_policy_coords)
                        eval_policy_coords[str(eval_point)] = {"coords": shape_orient_policy_coords, "difficulty": eval_difficulty_success_rate}
                    plot_success_rate_by_difficulty(eval_policy_coords,policy_eval_points, start_episode, eval_freq, max_episode, orientation, shape_name, variation_input_name, saving_dir=policy_filepath)

        else:
            # % TODO: Take in just the final 'best' policy for plotting success rate per difficulty
            policy_filepath += "/policy/pre-train_DDPGfD_kinovaGrip"
