import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# logger.addHandler(logging.FileHandler('/home/jovyan/work/rl-stuff/terminal_logs/evaluating_more_pretrains_aug17.log', 'a'))
#logger.addHandler(logging.FileHandler('/home/jovyan/work/rl-stuff/terminal_logs/half_speed_pretrain1k_train2k.log', 'a'))
logger.addHandler(logging.FileHandler('C:/Users/vanil/OSU_Robotics_GitHub/KinovaGrasping/gym-kinova-gripper/experiments/terminal_logs/half_speed_variable_speed.log', 'a'))
#logger.addHandler(logging.FileHandler('/scratch/hugheste/Steph_KinovaGrasping/01_Reinforcement_Learning/Pre_Training/Sim2Real_New_Grasp_Check/KinovaGrasping/gym-kinova-gripper/experiments/terminal_logs/half_speed_pretrain_train_new_grasp_check.log', 'a'))
print = lambda *x: logger.info("".join(str(item) for item in x))
print('============================================================================================================================ NEW_FILE =============================================================================================================================================================')

import numpy as np
import torch
import gym
import argparse
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import utils
#import TD3
#import OurDDPG
#import DDPG
import DDPGfD
import pdb
from tensorboardX import SummaryWriter
from ounoise import OUNoise
import pickle
import datetime
import csv
import timer
from timer import Timer
from pathlib import Path
import json # Allows us to read/write dictionaries
import copy # For copying over coordinates
import glob # Used for getting saved policy filename
from expert_data import ExpertPIDController, get_action # Expert controller, used within evaluation

# Import plotting code from other directory
plot_path = os.getcwd() + "/plotting_code"
sys.path.insert(1, plot_path)
from heatmap_plot import generate_heatmaps, create_heatmaps, overlap_images
from boxplot_plot import generate_reward_boxplots
from heatmap_coords import sort_and_save_heatmap_coords, sort_coords_by_region
from evaluation_plots import reward_plot, generate_heatmaps_by_orientation_frame
from replay_stats_plot import get_selected_episode_metrics, actual_values_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

import numpy as np
from scipy.spatial.transform import Rotation as R


def rotate_around_origin(vec, how_much_rotation):
    # vec: xyz
    # how_much_rotation: in radians. use np.radians to covert if necessary
    # roll, pitch, yaw.
    rotation = R.from_euler('xyz', how_much_rotation)

    rotated_vec = rotation.apply(vec)
    return rotated_vec


def rotate_around_sphere_mucrappo(rotvec_noise, local_rotation_pos_shift=np.array([0, 0, 0]),
                                  default_orientation=np.array([0, 0, 0]),
                                  grasp_translation=np.array([0, 0.18, 0.0654]),
                                  grasp_orientation=np.array([-1.57, 0, -1.57]),
                                  post_rotation_pos_shift=np.array([0, 0, 0])):
    """
    rotates a point around a sphere

    rotvec_noise: rotation noise to be apply extrinsically
    default_orientation: the default orientation. should stay at 0s
    grasp_translation: mujoco's euler value for whatever grasp position you're using
    grasp_orientation: mujoco's pos value for whatever grasp position you're using
    local_rotation_pos_shift: shifting the hand in local coordinates before the shift
    post_rotation_pos_shift: shifting the hand in global coordinates, after the inner shift...
    """
    # get rotation objects from starting grasp orientation, and the default orientation (should be 0 0 0)
    grasp_orientation_rotation = R.from_euler(seq='XYZ', angles=grasp_orientation,
                                              degrees=False)  # caps because instrinsic rotation
    default_orientation_rotation = R.from_euler(seq='XYZ', angles=default_orientation, degrees=False)

    # apply the orientation onto each other
    rotation_orientation = np.matmul(default_orientation_rotation.as_matrix(), grasp_orientation_rotation.as_matrix())

    # apply rotation noise
    desired_noise_rotation = R.from_rotvec(rotvec_noise)
    noisy_ori = np.matmul(desired_noise_rotation.as_matrix(), rotation_orientation)
    noisy_mujoco_euler = R.from_matrix(noisy_ori).as_euler(seq='XYZ')

    # get new postition
    noisy_mujoco_pos = rotate_around_origin(grasp_translation + local_rotation_pos_shift,
                                            rotvec_noise) + post_rotation_pos_shift

    return list(noisy_mujoco_euler), list(noisy_mujoco_pos)


def compare_test():
    """ Compare policy performance """
    #     eval_env = gym.make(env_name)
    #     eval_env.seed(seed + 100)
    #
    #     print("***Eval In Compare")
    #     # Generate randomized list of objects to select from
    #     eval_env.Generate_Latin_Square(eval_episodes,"eval_objects.csv",shape_keys=requested_shapes)
    #     ep_count = 0
    #     avg_reward = 0.0
    #     # step = 0
    #     for i in range(40):
    #         if i<23:
    #             x=(i)*0.005-0.055
    #             y=0.0
    #         elif i>=23:
    #             x=(i-23)*0.005-0.04
    #             y=0.0
    #         print('started pos', i)
    #         cumulative_reward = 0
    #         #eval_env = gym.make(env_name)
    #         state, done = eval_env.reset(shape_keys=requested_shapes, with_grasp=args.with_grasp_reward,start_pos=[x,y],env_name="eval_env", hand_orientation=requested_orientation,mode=mode), False
    #         # Set whether or not to use grasp reward
    #         eval_env.set_with_grasp_reward(args.with_grasp_reward)
    #         success=0
    #         # Sets number of timesteps per episode (counted from each step() call)
    #         #eval_env._max_episode_steps = 200
    #         eval_env._max_episode_steps = max_num_timesteps
    #         while not done:
    #             action = GenerateTestPID_JointVel(np.array(state),eval_env)
    #             env.set_with_grasp_reward(args.with_grasp_reward)
    #             state, reward, done, _ = eval_env.step(action)
    #             avg_reward += reward
    #             cumulative_reward += reward
    #             if reward > 25:
    #                 success=1
    #         num_success[1]+=success
    #         print('PID net reward:',cumulative_reward)
    #         state, done = eval_env.reset(shape_keys=requested_shapes, with_grasp=args.with_grasp_reward,start_pos=[x,y],env_name="eval_env", hand_orientation=requested_orientation,mode=mode), False
    #         # Set whether or not to use grasp reward
    #         eval_env.set_with_grasp_reward(args.with_grasp_reward)
    #         success=0
    #         cumulative_reward = 0
    #         # Sets number of timesteps per episode (counted from each step() call)
    #         #eval_env._max_episode_steps = 200
    #         eval_env._max_episode_steps = max_num_timesteps
    #         # Keep track of object coordinates
    #         obj_coords = eval_env.get_obj_coords()
    #         ep_count += 1
    #         print("***Eval episode: ",ep_count)
    #
    #         timestep = 0
    #         while not done:
    #             timestep += 1
    #             action = policy.select_action(np.array(state[0:82]))
    #             env.set_with_grasp_reward(args.with_grasp_reward)
    #             state, reward, done, _ = eval_env.step(action)
    #             avg_reward += reward
    #             cumulative_reward += reward
    #             if reward > 25:
    #                 success=1
    #             # eval_env.render()
    #             # print(reward)
    #         # pdb.set_trace()
    #         print('Policy net reward:',cumulative_reward)
    #         num_success[0]+=success
    #         print("Eval timestep count: ",timestep)
    #
    #         # Save initial object coordinate as success/failure
    #         x_val = (obj_coords[0])
    #         y_val = (obj_coords[1])
    #         x_val = np.asarray(x_val).reshape(1)
    #         y_val = np.asarray(y_val).reshape(1)
    #
    #         if (success):
    #             seval_obj_posx = np.append(seval_obj_posx, x_val)
    #             seval_obj_posy = np.append(seval_obj_posy, y_val)
    #         else:
    #             feval_obj_posx = np.append(feval_obj_posx, x_val)
    #             feval_obj_posy = np.append(feval_obj_posy, y_val)
    #
    #         total_evalx = np.append(total_evalx, x_val)
    #         total_evaly = np.append(total_evaly, y_val)
    #
    #     avg_reward /= eval_episodes
    #
    #     print("---------------------------------------")
    #     # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    #     print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
    #     print("---------------------------------------")

def evaluate_coords_by_region(policy, all_hand_object_coords, variation_type, all_saving_dirs, velocities, sample_size=2, regions_of_interest=None, controller_type="policy", state_idx_arr=np.arange(82)):
    """ Evaluate the policy within certain regions within the graspable area within the hand. Regions within
    the hand are determined by the Local coordinate frame. Plot and render a sample of success/failed coordinates.
    Policy: Policy to evaluate
    all_hand_object_coords: Coordinate dictionary with their representation within the local and global frame along with the
    hand orientation and evaluation results (reward value, success)
    variation_type: Input data type (Baseline, Baseline_HOV, etc.)
    variation_saving_dirs["heatmap_dir"]: Directory where the plots will be saved (Within the variation input directory)
    sample_size: Number of points to sample per region
    regions_of_interest: A list of the specific regions within the hand we would like to examine (Ex: ["extreme_left","extreme_right"])
    """
    frame = "local"
    requested_orientation = variation_type["requested_orientation"]
    if requested_orientation == "random":
        orientations_list = ["normal", "rotated", "top"]
    else:
        orientations_list = [requested_orientation]

    for orientation in orientations_list:
        heatmap_orient_dir = variation_saving_dirs["heatmap_dir"] + orientation + "/"
        all_shapes = [d["shape"] for d in all_hand_object_coords]
        for shape in all_shapes:
            heatmap_shape_dir = heatmap_orient_dir + shape + "/"

            hand_object_coords_dicts = [d for d in all_hand_object_coords if d["orientation"] == orientation]
            if regions_of_interest is None or regions_of_interest == "all_regions":
                grasping_regions = ["extreme_left", "mid_left", "center", "mid_right", "extreme_right"]
            else:
                grasping_regions = regions_of_interest

            for region_name in grasping_regions:
                # Divide the coordinates by region
                region_dicts = [d for d in hand_object_coords_dicts if d["local_coord_region"] == region_name]
                if len(region_dicts) > 0:
                    region_dir = heatmap_shape_dir + "regions/" + region_name + "/"
                    create_paths([region_dir])

                    # Sample successful and failed points from evaluation
                    success_dicts = [d for d in region_dicts if d["success"] is True]
                    fail_dicts = [d for d in region_dicts if d["success"] is False]

                    wrist_coords = None
                    finger_coords = None

                    # Sample points from each region of success/failures to render
                    for dict_list in [success_dicts, fail_dicts]:
                        num_points = min(sample_size,len(dict_list))
                        for d_idx in range(num_points):
                            curr_dict = dict_list[d_idx]

                            # Evaluate policy with a sampled point in a region and create a video rendering
                            region_eval_ret = eval_policy(policy, args.env_name, args.seed,
                                                          requested_shapes=variation_type["requested_shapes"],
                                                          requested_orientation=curr_dict["orientation"],
                                                          eval_episodes=1,
                                                          render_imgs=True,
                                                          start_pos=curr_dict["global_obj_coords"],
                                                          hand_rotation=curr_dict["hand_orient_variation"],
                                                          all_saving_dirs=all_saving_dirs,
                                                          velocities=velocities,
                                                          output_dir=region_dir,
                                                          with_noise=variation_type["with_orientation_noise"],
                                                          controller_type=controller_type,
                                                          max_num_timesteps=max_num_timesteps, state_idx_arr=state_idx_arr)

                            all_action_values = region_eval_ret["all_action_values"]
                            controller_actions = all_action_values["controller_actions"]
                            render_file_dir = region_eval_ret["render_file_dir"]
                            range_of_episodes = np.arange(len(controller_actions))

                    # Save the hand and object coordinates
                    if len(success_dicts) > 0:
                        dict_file = open(region_dir + "/success_coords_dicts.csv", "w", newline='')
                        keys = success_dicts[0].keys()
                        dict_writer = csv.DictWriter(dict_file, keys)
                        dict_writer.writeheader()
                        dict_writer.writerows(success_dicts)
                        dict_file.close()

                    if len(fail_dicts) > 0:
                        dict_file = open(region_dir + "/fail_coords_dicts.csv", "w", newline='')
                        keys = fail_dicts[0].keys()
                        dict_writer = csv.DictWriter(dict_file, keys)
                        dict_writer.writeheader()
                        dict_writer.writerows(fail_dicts)
                        dict_file.close()

                    # Just take out the successful local coordinates for plotting
                    success_coords = [d[frame + "_obj_coords"] for d in region_dicts if d["success"] is True]
                    fail_coords = [d[frame + "_obj_coords"] for d in region_dicts if d["success"] is False]

                    success_x = [coords[0] for coords in success_coords]
                    success_y = [coords[1] for coords in success_coords]
                    fail_x = [coords[0] for coords in fail_coords]
                    fail_y = [coords[1] for coords in fail_coords]
                    total_x = success_x + fail_x
                    total_y = success_y + fail_y

                    create_heatmaps(success_x, success_y, fail_x, fail_y, total_x, total_y, shape, orientation,
                                    state_rep=frame, saving_dir=region_dir, wrist_coords=wrist_coords, finger_coords=finger_coords,
                                    title_str="Input Variation: " + variation_type["variation_name"] + ", " + frame.capitalize() + " coord. frame, Region: " + region_name.capitalize())

def local_to_global_transf(local_coords, Tfw):
    """ Convert coordinates from the local frame to the global frame """
    # Local to Global coordinate conversion (from our observation)
    global_temp = np.append(local_coords, 1)
    global_temp = np.matmul(np.linalg.inv(Tfw), global_temp)
    local_to_global_coords = global_temp[0:3].tolist()
    return local_to_global_coords

def test_global_to_local_transformation(local_coords,global_coords,env_global_coords, Tfw):
    """ Get the current observation and the environment's global --> local transformation matrix and check
    the output is the same.
    """
    # Number of decimal places to test against
    precision = 10

    # Global to Local coordinate conversion (from our observation)
    local_temp = np.append(global_coords, 1)
    local_temp = np.matmul(Tfw, local_temp)
    local_temp_coords = local_temp[0:3].tolist()

    # Check our observed global coords, transformed to the local frame, match the observed local coordinates
    np.testing.assert_array_almost_equal(local_temp_coords,local_coords, decimal=precision, err_msg="Observed Global coords, transformed to Local, do NOT equal the observed Local coordinates")

    # Local to Global coordinate conversion (from our observation)
    global_temp = np.append(local_coords, 1)
    global_temp = np.matmul(np.linalg.inv(Tfw), global_temp)
    global_temp_coords = global_temp[0:3].tolist()

    # Check our observed local coords, transformed to the global frame, match the observed global coordinates
    np.testing.assert_array_almost_equal(global_temp_coords, global_coords, decimal=precision, err_msg="Observed Local coords, transformed to Global, do NOT equal the observed Global coordinates")

    # Check our observed global coordinates match what the environment has set for the global coordinates (should match)
    if env_global_coords is not None:
        np.testing.assert_array_almost_equal(global_coords, env_global_coords, decimal=precision,
                                      err_msg="Observed Global coords do NOT equal what the environment has set for the Global coordinates")

    print("Done with test_global_to_local_transformation, (decimal precision: "+str(precision)+", all test cases PASSED! :)")


def check_grasp(f_dist_old, f_dist_new, total_distal_change):
    """
    Uses the current change in x,y position of the distal finger tips, summed over all fingers to determine if
    the object is grasped (fingers must have only changed in position over a tiny amount to be considered done).
    f_dist_old: Distal finger tip x,y,z coordinate values from previous timestep
    f_dist_new: Distal finger tip x,y,z coordinate values from current timestep
    total_distal_change: cumulative total change of the distal fingertips over the course of the episode
    """

    # Initial check to see if previous state has been set
    if f_dist_old is None:
        return False, total_distal_change
    sampling_time = 15

    # Change in finger 1 distal x-coordinate position
    f1_change = abs(f_dist_old[0] - f_dist_new[0])
    f1_diff = f1_change / sampling_time

    # Change in finger 2 distal x-coordinate position
    f2_change = abs(f_dist_old[3] - f_dist_new[3])
    f2_diff = f2_change / sampling_time

    # Change in finger 3 distal x-coordinate position
    f3_change = abs(f_dist_old[6] - f_dist_new[6])
    f3_diff = f3_change / sampling_time

    # Sum of changes in distal fingers
    f_all_change = f1_diff + f2_diff + f3_diff

    total_distal_change["finger_1"] += f1_diff
    total_distal_change["finger_2"] += f2_diff
    total_distal_change["finger_3"] += f3_diff

    #print("In check grasp, f1_change: {}, f2_change: {}, f3_change: {}, f_all_change: {}".format(f1_change,f2_change,f3_change,f_all_change))

    # If the fingers have only changed a small amount, we assume the object is grasped
    if f_all_change < 0.0001 and total_distal_change["finger_1"] > 0.0001 and total_distal_change["finger_2"] > 0.0001 and total_distal_change["finger_3"] > 0.0001:
        return True, total_distal_change
    else:
        return False, total_distal_change


def get_hand_object_coords_dict(curr_env):
    """Create dictionary containing info about the hand-object pose from a single episode"""
    # Global object and hand coordinates
    hand_object_coords = {"local_obj_coords": [], "global_obj_coords": [], "local_to_global_obj_coords": [],
                          "local_wrist_coords": [], "global_wrist_coords": [], "local_to_global_wrist_coords": [],
                          "global_to_local_transf": [], "shape": [], "orientation": [], "hand_orient_variation": [],
                          "local_finger_coords": [], "global_finger_coords": [], "local_to_global_finger_coords": [],
                          "local_coord_region": "None", "total_episode_reward": 0, "success": 0}
    global_to_local_transf = curr_env.Tfw
    hand_orient_variation = curr_env.hand_orient_variation

    coords_file = curr_env.get_coords_filename()
    coords_file_idx = curr_env.get_orientation_idx()

    # Local finger positions (within the state)
    local_state = curr_env.get_obs_from_coord_frame(coord_frame="local")
    global_to_local_transf = curr_env.Tfw
    local_finger_coords = local_state[0:18]  # Finger 1, 2, 3 proximal (x,y,z) coords, followed by distal coords
    local_wrist_coords = local_state[18:21]
    local_obj_coords = local_state[21:24]

    # Global hand and object positions
    global_state = curr_env.get_obs_from_coord_frame(coord_frame="global")
    global_to_local_transf = curr_env.Tfw
    global_finger_coords = global_state[0:18]
    global_wrist_coords = global_state[18:21]
    global_obj_coords = global_state[21:24]

    orientation = curr_env.get_orientation()
    shape = curr_env.get_random_shape()

    # Save the hand-object coordinates to track the transformations
    hand_object_coords["local_obj_coords"] = local_obj_coords
    hand_object_coords["global_obj_coords"] = global_obj_coords
    hand_object_coords["local_wrist_coords"] = local_wrist_coords
    hand_object_coords["local_finger_coords"] = local_finger_coords
    hand_object_coords["global_finger_coords"] = global_finger_coords
    hand_object_coords["global_wrist_coords"] = global_wrist_coords
    hand_object_coords["global_to_local_transf"] = global_to_local_transf
    hand_object_coords["orientation"] = orientation
    hand_object_coords["shape"] = shape
    hand_object_coords["hand_orient_variation"] = hand_orient_variation
    hand_object_coords["coords_file"] = coords_file
    hand_object_coords["coords_file_idx"] = coords_file_idx

    # Local to Global coordinate conversion (from our observation)
    local_to_global_temp = np.append(local_obj_coords, 1)
    local_to_global_temp = np.matmul(np.linalg.inv(global_to_local_transf), local_to_global_temp)
    local_to_global_obj_coords = local_to_global_temp[0:3].tolist()

    hand_object_coords["local_to_global_wrist_coords"] = local_to_global_transf(local_wrist_coords,
                                                                                global_to_local_transf)
    hand_object_coords["local_to_global_obj_coords"] = local_to_global_transf(local_obj_coords, global_to_local_transf)
    for coord_idx in range(0, len(local_finger_coords), 3):
        transf_coords = local_to_global_transf(local_finger_coords[coord_idx:coord_idx + 3], global_to_local_transf)
        hand_object_coords["local_to_global_finger_coords"].extend(transf_coords)

    return hand_object_coords

# Runs policy for X episodes and returns average reward -- evaluate the policy per shape and per hand orientation
def eval_policy(policy, env_name, seed, requested_shapes, requested_orientation, with_noise, controller_type, max_num_timesteps, all_saving_dirs, velocities, n_steps=5, output_dir=None, start_pos=None,hand_rotation=None,eval_episodes=100, compare=False, render_imgs=True, state_idx_arr=np.arange(82)):
    """ Evaluate policy in its given state over eval_episodes amount of grasp trials """
    num_success=0
    # Initial (timestep = 1) transformation matrices (from Global to Local) for each episode
    all_hand_object_coords = []

    # Compare policy performance
    if compare:
        compare_test()

    # Make new environment for evaluation
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    # Generate randomized list of objects to select from
    eval_env.Generate_Latin_Square(eval_episodes,"eval_objects.csv",shape_keys=requested_shapes)

    cumulative_reward = 0 # Total reward over all episodes
    # Reward data over each evaluation episode for boxplot
    all_ep_reward_values = {"total_reward": [], "finger_reward": [], "grasp_reward": [], "lift_reward": []}
    all_action_values = {"controller_actions":[]}

    if output_dir is None:
        output_dir = all_saving_dirs["output_dir"]

    # directory where the simulation renderings are stored
    render_file_dir = output_dir

    for i in range(eval_episodes):
        print("***Eval episode: ", i)
        state, done = eval_env.reset(shape_keys=requested_shapes, with_grasp=args.with_grasp_reward,hand_orientation=requested_orientation,mode="eval",env_name="eval_env",orient_idx=i,with_noise=with_noise,start_pos=start_pos,hand_rotation=hand_rotation), False

        # Initialize the controller if the controller type is not a policy
        if controller_type != "policy":
            controller = ExpertPIDController(state)

        # Record initial coordinate file path once shapes are generated
        all_saving_dirs["eval_init_coord_file_path"] = eval_env.get_coords_filename()
        # Sets number of timesteps per episode (counted from each step() call)
        eval_env._max_episode_steps = max_num_timesteps

        # Record the hand-object pose within a dict for plotting and saving
        hand_object_coords = get_hand_object_coords_dict(eval_env)

        # Cumulative reward over single episode
        ep_finger_reward = 0
        ep_grasp_reward = 0
        ep_lift_reward = 0

        # Actions taken by the policy over an episode
        episode_actions = []
        episode_reward = 0 # Cumulative reward over a single episode
        final_grasp_reward = 0 # Used to determine the final grasp reward value (that will be replaced)
        ready_for_lift = False # Signals if we are ready for lifting, initially false as we have not moved the hand
        lift_success = 0
        total_distal_change = {"finger_1": 0, "finger_2": 0, "finger_3": 0,} # Cumulative change in the distal finger tips over the course of the episode

        # Beginning of episode time steps, done is max time steps or lift reward achieved
        timestep = 1
        lift_timestep = 1

        # Grasping Stage
#         while timestep <= 30 and not ready_for_lift:
        while timestep <= 45 and not ready_for_lift:  # ADAM: INCREASE TO 45 TIMESTEPS
            if controller_type == "policy":
                finger_action = policy.select_action(np.array(state[0:82])[state_idx_arr])
            else:
                # Get the action from the controller (controller_type: naive, position-dependent)
                finger_action = get_action(obs=np.array(state[0:82])[state_idx_arr], lift_check=ready_for_lift, controller=controller, env=eval_env, velocities=velocities, pid_mode=controller_type, timestep=timestep)

            wrist_action = np.array([0])
            action = np.concatenate((wrist_action, finger_action)) # If not ready for lift, wrist should always be 0

            next_state, reward, grasp_done, info = eval_env.step(action) # Step action takes in the wrist velocity plus the finger velocities

            episode_reward += reward
            final_grasp_reward = reward # Used to determine the final grasp reward value (that will be replaced)

            # Cumulative reward
            ep_finger_reward += info["finger_reward"]
            ep_grasp_reward += info["grasp_reward"]
            ep_lift_reward += info["lift_reward"]

            # Used for plotting finger actions
            episode_actions.append(finger_action)

            # Render the performance
            if render_imgs is True:
                action_str = "Grasping stage timestep: "+str(timestep)+"\nAction (rad/sec):\nWrist Velocity: {}\nFinger 1 Velocity: {:.3f}\nFinger 2 Velocity: {:.3f}\nFinger 3 Velocity: {:.3f}".format(action[0], action[1], action[2], action[3])
                action_str = action_str + "\nObject Position (local x,y,z): {:.3f}, {:.3f}, {:.3f}\nReward: {}".format(hand_object_coords["local_obj_coords"][0], hand_object_coords["local_obj_coords"][1], hand_object_coords["local_obj_coords"][2], reward)
                render_file_dir = eval_env.render_img(text_overlay=action_str, episode_num=i, timestep_num=timestep,obj_coords=hand_object_coords["local_obj_coords"], saving_dir=output_dir,final_episode_type=None)

            # Set the previous state and the current state distal finger tips positions
            f_dist_old = state[9:17]
            f_dist_new = next_state[9:17]

            # Check if the movement in the distal finger tips give a grasping position
            if timestep >= n_steps:
                ready_for_lift, total_distal_change = check_grasp(f_dist_old, f_dist_new, total_distal_change)

            state = next_state
            timestep = timestep + 1

        # Lifting stage
        while lift_timestep <= 15 and not done:
            # Lift the hand with the pre-determined lifting velocities
            action = np.array([velocities["wrist_lift_velocity"], velocities["finger_lift_velocity"], velocities["finger_lift_velocity"], velocities["finger_lift_velocity"]])
            next_state, reward, done, info = eval_env.step(action)

            # Done in the lifting stage is determined by whether we reach the target lifting height of the object
            if done:
                # Replace the final reward value with the lift reward
                episode_reward = episode_reward - final_grasp_reward + reward

                # Determine success of the episode based on the final lift reward
                if reward == 50:
                    lift_success = 1
                # Cumulative reward per reward type
                ep_finger_reward += info["finger_reward"]
                ep_grasp_reward += info["grasp_reward"]
                ep_lift_reward += info["lift_reward"]

            if render_imgs is True:
                action_str = "Lifting stage timestep: "+str(lift_timestep)+"\nConstant lift action by controller"+"\nAction (rad/sec):\nWrist Velocity: {}\nFinger 1 Velocity: {:.3f}\nFinger 2 Velocity: {:.3f}\nFinger 3 Velocity: {:.3f}".format(action[0], action[1], action[2], action[3])
                action_str = action_str + "\nObject Position (local x,y,z): {:.3f}, {:.3f}, {:.3f}\nReward: {}".format(hand_object_coords["local_obj_coords"][0], hand_object_coords["local_obj_coords"][1], hand_object_coords["local_obj_coords"][2], reward)
                render_file_dir = eval_env.render_img(text_overlay=action_str, episode_num=i, timestep_num=(timestep+lift_timestep-1),obj_coords=hand_object_coords["local_obj_coords"], saving_dir=output_dir,final_episode_type=None)

            state = next_state
            lift_timestep = lift_timestep + 1

        ## End of the episode ###
        # Cumulative reward over all episodes
        cumulative_reward += episode_reward

        # Record findings
        all_ep_reward_values["total_reward"].append(episode_reward)
        all_ep_reward_values["finger_reward"].append(ep_finger_reward)
        all_ep_reward_values["grasp_reward"].append(ep_grasp_reward)
        all_ep_reward_values["lift_reward"].append(ep_lift_reward)

        # Record all actions taken by the policy (with and without noise)
        if len(episode_actions) > 0:
            episode_actions = np.stack(episode_actions, axis=0)

        all_action_values["controller_actions"].append(episode_actions)

        num_success += lift_success

        # Move the rendered images to their final directory location based on success/failure and plot the velocities from the episode
        if render_imgs is True:
            if lift_success != 1:
                print("Failed episode!!")
            render_file_dir = eval_env.render_img(text_overlay=action_str, episode_num=i,
                                                  timestep_num=(timestep + lift_timestep - 1),
                                                  obj_coords=hand_object_coords["local_obj_coords"], saving_dir=output_dir,
                                              final_episode_type=lift_success)
            for metric_idx in range(0, 3):
                metric_actions = [action_value[metric_idx] for action_value in episode_actions]
                axes_limits = {"x_min": 0, "x_max": len(metric_actions), "y_min": 0, "y_max": velocities["max_velocity"]}
                actual_values_plot([metric_actions], 0, "Finger " + str(metric_idx + 1) + " Velocity",
                                   "Action Output: Finger " + str(metric_idx + 1) + " Velocity",
                                   axes_limits=axes_limits, saving_dir=render_file_dir)

        hand_object_coords["total_episode_reward"] = episode_reward
        hand_object_coords["success"] = bool(lift_success)
        all_hand_object_coords.append(hand_object_coords)

    # Determine the average reward over all evaluation episodes
    avg_reward = cumulative_reward / eval_episodes

    # Final average reward values over all episodes
    avg_rewards = {}
    avg_rewards["total_reward"] = np.average(all_ep_reward_values["total_reward"])
    avg_rewards["finger_reward"] = np.average(all_ep_reward_values["finger_reward"])
    avg_rewards["grasp_reward"] = np.average(all_ep_reward_values["grasp_reward"])
    avg_rewards["lift_reward"] = np.average(all_ep_reward_values["lift_reward"])

    sort_coords_by_region(all_hand_object_coords)  # Fills in the coordinate region

    print("---------------------------------------")
    print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
    print("---------------------------------------")

    ret = {"avg_reward": avg_reward, "avg_rewards": avg_rewards, "all_ep_reward_values": all_ep_reward_values, "all_action_values" : all_action_values, "num_success": num_success, "all_hand_object_coords": all_hand_object_coords, "render_file_dir": render_file_dir}
    return ret


def write_tensor_plot(writer,episode_num,avg_reward,avg_rewards,actor_loss,critic_loss,critic_L1loss,critic_LNloss, current_expert_proportion):
    """ Write important data about policy performance to tensorboard Summary Writer
    writer: Tensorboard Summary Writer
    episode_num: Current episode number
    avg_reward: Reward average over evaulation grasp trials
    avg_rewards: Dictionary of average reward values for Finger, Grasp, and Lift rewards over evaluation grasp trials
    actor_loss,critic_loss,critic_L1loss,critic_LNloss
    """
    
    writer.add_scalar("Episode total reward, Avg. " + str(args.eval_freq) + " episodes", avg_reward, episode_num)
    writer.add_scalar("Episode finger reward, Avg. " + str(args.eval_freq) + " episodes", avg_rewards["finger_reward"],
                      episode_num)
    writer.add_scalar("Episode grasp reward, Avg. " + str(args.eval_freq) + " episodes",
                      avg_rewards["grasp_reward"], episode_num)
    writer.add_scalar("Episode lift reward, Avg. " + str(args.eval_freq) + " episodes",
                      avg_rewards["lift_reward"], episode_num)
    writer.add_scalar("Actor loss", actor_loss, episode_num)
    writer.add_scalar("Critic loss", critic_loss, episode_num)
    writer.add_scalar("Critic L1loss", critic_L1loss, episode_num)
    writer.add_scalar("Critic LNloss", critic_LNloss, episode_num)
    writer.add_scalar("Expert Sampling Proportion", current_expert_proportion, episode_num)

    return writer


def conduct_episodes(policy, controller_type, expert_buffers, replay_buffer, num_episodes, prob, type_of_training, all_saving_dirs, max_num_timesteps, velocities, state_idx_arr=np.arange(82)):
    """ Conduct the desired number of episodes with the controller
    policy: policy to be updated
    expert_buffers: dictionary of expert replay buffers, where the key is the current shape and the value is the expert replay buffer for that shape
    replay_buffer: agent replay buffer
    num_episodes: Max number of episodes to update over
    prob: Probability (proportion) of sampling from expert replay buffer
    type_of_training: Based on training mode ("pre-train", "test", etc.)
    max_num_timesteps: Maximum number of time steps within a RL episode
    state_idx_arr: Modifies the state dimensionality.
    """

    print('======================== CONDUCT EPISODES - STATE INDEX ARRAY SIZE: ', len(state_idx_arr), ' ==============================')

    # Initialize and copy current policy to be the best policy
    best_policy = DDPGfD.DDPGfD(state_dim=len(state_idx_arr))  # TODO: just pass in a whole args... bruh
    best_policy.copy(policy)

    # Initialize OU noise, added to action output from policy
    noise = OUNoise(4)
    noise.reset()
    expl_noise = OUNoise(4, sigma=0.001)
    expl_noise.reset()

    # CONTROLLER: Collect all initial object coordinates sorted by success/failure for heatmap plotting
    all_hand_object_coords = []
    
    # Number of successful/failed initial object coordinates from evaluation over the total # of grasp trials
    num_success = 0
    num_fail = 0

    # Stores reward boxplot data, Average reward per evaluation episodes
    finger_reward = [[]]
    grasp_reward = [[]]
    lift_reward = [[]]
    total_reward = [[]]
    eval_action = [[]]

    # All policy actions over each episode
    all_action_ouput = {"episode": [], "timestep": [], "action": [], "policy_action": [], "action_noise": []}

    # All policy loss output at each evaluation frequency
    all_policy_output = {"actor_loss": [], "critic_loss": [], "critic_L1loss": [], "critic_LNloss": []}

    # Tensorboard writer
    if controller_type == "policy":
        writer = SummaryWriter(logdir=all_saving_dirs["tensorboard_dir"])

    env = gym.make(args.env_name)
    # Max number of time steps to match the expert replay grasp trials
    env._max_episode_steps = max_num_timesteps
    episode_num = 0 # Current number of episodes

    for _ in range(num_episodes):
        # Fill training object list using latin square
        if env.check_obj_file_empty("objects.csv"):
            env.Generate_Latin_Square(args.max_episode, "objects.csv", shape_keys=requested_shapes)

        # Reset the environment
        state, done = env.reset(shape_keys=requested_shapes, with_grasp=args.with_grasp_reward,env_name="env", hand_orientation=requested_orientation,
                                mode=args.mode, with_noise=with_orientation_noise), False

        # If we are not using a policy, intialize the controller
        if controller_type != "policy":
            controller = ExpertPIDController(state)

        # Set whether or not to use grasp reward
        env.set_with_grasp_reward(args.with_grasp_reward)
        # Record initial coordinate file path once shapes are generated
        all_saving_dirs["train_init_coord_file_path"] = env.get_coords_filename()

        noise.reset()
        expl_noise.reset()

        obj_coords = env.get_obj_coords() # Global object position coordinates
        obj_local = np.append(obj_coords,1) # Local object position coordinate conversion
        obj_local = np.matmul(env.Tfw,obj_local)
        local_obj_coords = obj_local[0:3]

        # Orientation HOV index to be recorded by replay buffer
        orientation_idx = env.get_orientation_idx()

        # Determine the expert replay buffer to be used based on the selected shape
        current_object = env.get_random_shape()
        # Only try to access the expert replay buffer if it exists
        if expert_buffers is not None:
            expert_replay_buffer = expert_buffers[current_object]
        else:
            expert_replay_buffer = None

        print(type_of_training, episode_num)

        # Record the hand-object pose within a dict for plotting and saving
        hand_object_coords = get_hand_object_coords_dict(env)

        episode_reward = 0 # Cumulative reward over a single episode
        ready_for_lift = False # Signals if we are ready for lifting, initially false as we have not moved the hand
        lift_success = 0
        total_distal_change = {"finger_1": 0, "finger_2": 0, "finger_3": 0,}  # Cumulative change in the distal finger tips over the course of the episode

        replay_buffer_recorded_ts = 0  # Number of RL time steps recorded by the replay buffer
        replay_buffer.add_episode(1)  # Start recording the episode within the replay buffer

        # Beginning of episode time steps, done is max time steps or lift reward achieved
        timestep = 1
        lift_timestep = 1

        # Grasping Stage
#         while timestep <= 30 and not ready_for_lift:
        while timestep <= 45 and not ready_for_lift:  # ADAM: Increase timestep to 45 (50%)
            replay_buffer.add_orientation_idx_to_replay(orientation_idx)

            if controller_type == "policy":
                policy_action = policy.select_action(np.array(state)[state_idx_arr])
                action_noise = np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                finger_action = (policy_action + action_noise).clip(0, max_action)

                # Record the policy's output action and the added noise within the replay buffer
                replay_buffer.policy_action[-1].append(policy_action)
                replay_buffer.action_noise[-1].append(action_noise)

                # Record the action output over all episodes to track the policy output
                all_action_ouput["episode"].append(episode_num)
                all_action_ouput["timestep"].append(timestep)
                all_action_ouput["policy_action"].append(policy_action)
                all_action_ouput["action_noise"].append(action_noise)
            else:
                # Get the action from the controller (controller_type: naive, position-dependent)
                finger_action = get_action(obs=np.array(state[0:82])[state_idx_arr], lift_check=ready_for_lift, controller=controller, env=env, velocities=velocities, pid_mode=controller_type)

            wrist_action = np.array([0])
            action = np.concatenate((wrist_action, finger_action))  # Wrist velocity will be 0 until ready for lift

            # Records all action output (even if it is not contained within the replay buffer)
            all_action_ouput["action"].append(action)

            # Perform grasping action by the policy
            next_state, reward, grasp_done, info = env.step(action)

            # Record the transition within the replay buffer
            replay_buffer.add(state[0:82], finger_action, next_state[0:82], reward, float(grasp_done))
            replay_buffer_recorded_ts += 1

            episode_reward += reward

            # Set the previous state and the current state distal finger tips positions
            f_dist_old = state[9:17]
            f_dist_new = next_state[9:17]

            # Check if the movement in the distal finger tips give a grasping position
            if timestep >= replay_buffer.n_steps:
                ready_for_lift, total_distal_change = check_grasp(f_dist_old, f_dist_new, total_distal_change)

            state = next_state
            timestep = timestep + 1

        replay_buffer.add_episode(0)  # Add entry for new episode

        # Lifting stage
        while lift_timestep <= 15 and not done:
            # Lift the hand with the pre-determined lifting velocities
            action = np.array([velocities["wrist_lift_velocity"], velocities["finger_lift_velocity"], velocities["finger_lift_velocity"], velocities["finger_lift_velocity"]])
            next_state, reward, done, info = env.step(action)

            # Done in the lifting stage is determined by whether we reach the target lifting height of the object
            if done:
                # Replace the final recorded reward within the replay buffer
                old_reward = replay_buffer.replace(reward, done)
                episode_reward = episode_reward - old_reward + reward # Add the final reward (replacing the old reward value)

                # Determine success of the episode based on the final lift reward
                if reward == 50:
                    lift_success = 1

            state = next_state
            lift_timestep = lift_timestep + 1

        # Record the outcome of the episode given the hand-object pose
        hand_object_coords["total_episode_reward"] = episode_reward
        hand_object_coords["success"] = bool(lift_success)
        all_hand_object_coords.append(hand_object_coords)

        # Remove any invalid episodes (episodes shorter than n-step length for policy training)
        episode_len = replay_buffer_recorded_ts # Number of timesteps within the episode recorded by replay buffer
        if episode_len == 0 or episode_len < replay_buffer.n_steps:
                replay_buffer.remove_episode(-1)  # If episode is invalid length (less that n-steps), remove it

        ## CONTROLLER: Track entire experience of success/failed grasp trials for heatmap plotting
        if controller_type != "policy":
            # If at the final episode, save all coordinates
            if episode_num + 1 == num_episodes:
                sort_and_save_heatmap_coords(all_hand_object_coords, requested_shapes, requested_orientation_list, episode_num=None, saving_dir=all_saving_dirs["heatmap_dir"])

                # Records the number of successful and failed coordinates over all episodes
                num_fail = args.max_episode - num_success

        ## POLICY TRAINING AND EVALUATION
        if controller_type == "policy":
            # Train agent after collecting sufficient data:
            if episode_num >= args.update_after: # Update policy after 100 episodes (have enough experience in agent replay buffer)
                for update_count in range(args.update_num): # Number of times to update the policy
                    if args.batch_size == 0:
                        # Single episode training using full trajectory
                        actor_loss, critic_loss, critic_L1loss, critic_LNloss = policy.train(env._max_episode_steps,
                                                                                         expert_replay_buffer,
                                                                                         replay_buffer, prob, mod_state_idx=state_idx_arr)
                    else:
                        # Batch training using n-steps
                        actor_loss, critic_loss, critic_L1loss, critic_LNloss = policy.train_batch(env._max_episode_steps, episode_num, update_count,
                                                                                             expert_replay_buffer,
                                                                                             replay_buffer, mod_state_idx=state_idx_arr)
            # Evaluation of the policy: Evaluate the policy every eval_freq episodes
            if episode_num+1 == num_episodes or (episode_num) % args.eval_freq == 0:
                print("EVALUATING EPISODE AT: ",episode_num)
                print("Evaluating with "+str(args.eval_num)+" grasping trials")
                eval_ret = eval_policy(policy, args.env_name, args.seed, requested_shapes, requested_orientation, velocities=velocities, all_saving_dirs=all_saving_dirs, max_num_timesteps=max_num_timesteps,
                                       controller_type=controller_type, eval_episodes=args.eval_num, with_noise=with_orientation_noise, render_imgs=args.render_imgs, state_idx_arr=state_idx_arr)

                # Records the number of successful and failed coordinates from evaluation
                num_success = eval_ret["num_success"]
                num_fail = args.eval_num - num_success

                # Filter through coordinates by success, failure, shape, and orientation, then save them
                eval_hand_object_coords = eval_ret["all_hand_object_coords"]
                if episode_num + 1 == num_episodes:
                    episode_num += 1    # Save the final episode coordinates

                # Sort coordinates by success/failure and same them by evaluation point
                print("Saving heatmap data at: ", all_saving_dirs["heatmap_dir"])
                sort_and_save_heatmap_coords(eval_hand_object_coords, requested_shapes, requested_orientation_list, episode_num=episode_num, saving_dir=all_saving_dirs["heatmap_dir"])

                # Cumulative (over timesteps) reward data from each evaluation episode for boxplot
                all_ep_reward_values = eval_ret["all_ep_reward_values"]
                all_eval_action_values = eval_ret["all_action_values"]

                # Plot tensorboard metrics for learning analysis (average reward, loss, etc.)
                all_policy_output["actor_loss"].append(policy.actor_loss)
                all_policy_output["critic_loss"].append(policy.critic_loss)
                all_policy_output["critic_L1loss"].append(policy.critic_L1loss)
                all_policy_output["critic_LNloss"].append(policy.critic_LNloss)
                writer = write_tensor_plot(writer,episode_num,eval_ret["avg_reward"],eval_ret["avg_rewards"],policy.actor_loss,policy.critic_loss,policy.critic_L1loss,policy.critic_LNloss,policy.current_expert_proportion)

                # Insert boxplot code reference
                finger_reward[-1].append(all_ep_reward_values["finger_reward"])
                grasp_reward[-1].append(all_ep_reward_values["grasp_reward"])
                lift_reward[-1].append(all_ep_reward_values["lift_reward"])
                total_reward[-1].append(all_ep_reward_values["total_reward"])
                eval_action[-1].append(all_eval_action_values["controller_actions"])

                # Save a copy of the current policy for evaluation purposes
                evaluated_policy_path = all_saving_dirs["results_saving_dir"] + "/policy_" + str(episode_num) + "/"
                create_paths([evaluated_policy_path])
                policy.save(evaluated_policy_path)
                print("Evaluation from "+str(episode_num)+"Saving current policy at: ",all_saving_dirs["results_saving_dir"]+"policy_"+str(episode_num)+"/")

                # Check if the current policy is the best policy
                policy.avg_evaluation_reward = eval_ret["avg_reward"]
                if policy.avg_evaluation_reward >= best_policy.avg_evaluation_reward:
                    best_policy.copy(policy)
                    print("Evaluation episode: "+str(episode_num)+" COPYING Current Policy to be the BEST policy")

            # Evaluation of the policy: Save evaluation data every save_freq episodes
            if episode_num+1 == num_episodes or (episode_num) % args.save_freq == 0:
                print("Saving boxplot data at: ", all_saving_dirs["boxplot_dir"])
                np.save(all_saving_dirs["boxplot_dir"] + "/finger_reward_" + str(episode_num),finger_reward)
                np.save(all_saving_dirs["boxplot_dir"] + "/grasp_reward_" + str(episode_num),grasp_reward)
                np.save(all_saving_dirs["boxplot_dir"] + "/lift_reward_" + str(episode_num),lift_reward)
                np.save(all_saving_dirs["boxplot_dir"] + "/total_reward_" + str(episode_num),total_reward)

                # Save the finger velocities for plotting and evaluation
                actions_path = all_saving_dirs["results_saving_dir"] + "/eval_actions/eval_controller_actions_" + str(episode_num)
                create_paths([all_saving_dirs["results_saving_dir"] + "/eval_actions/"])
                np.save(actions_path, all_eval_action_values["controller_actions"])

                # Save all action output - overwrite what is there so in the end it is a full list
                dict_file = open(all_saving_dirs["results_saving_dir"] + "/all_action_output.csv", "w", newline='')
                keys = all_action_ouput.keys()
                dict_writer = csv.DictWriter(dict_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows([all_action_ouput])
                dict_file.close()

                # Save all policy loss output - overwrite what is there so in the end it is a full list
                dict_file = open(all_saving_dirs["results_saving_dir"] + "/all_policy_output.csv", "w", newline='')
                keys = all_policy_output.keys()
                dict_writer = csv.DictWriter(dict_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows([all_policy_output])
                dict_file.close()

                finger_reward = [[]]
                grasp_reward = [[]]
                lift_reward = [[]]
                total_reward = [[]]
                eval_action = [[]]

        episode_num += 1

    # Training is complete, now save policy and replay buffer
    if controller_type == "policy":
        # Save policy
        print("Saving the BEST policy...")
        best_policy.save(all_saving_dirs["model_save_path"])
        print("Saved policy at: ", all_saving_dirs["model_save_path"])

    print("Saving Agent replay buffer experience...")
    replay_buffer.save_replay_buffer(all_saving_dirs["replay_buffer"])
    print("Saved agent replay buffer at: ", all_saving_dirs["replay_buffer"])

    return num_success, num_fail


def create_paths(dir_list):
    """ Create directories if they do not exist already, given path """
    for new_dir in dir_list:
        if new_dir is not None:
            new_path = Path(new_dir)
            new_path.mkdir(parents=True, exist_ok=True)


def setup_directories(env, saving_dir, expert_replay_file_path, agent_replay_file_path, pretrain_model_save_path, create_dirs=True):
    """ Setup directories where information will be saved
    env: Pass in current environment to have access to getting environment variables for recording purposes
    saving_dir: main name for all related files (ex: train_DDPGfD_CubeS)
    expert_replay_file_path: Expert replay buffer file path
    agent_replay_file_path: Agent replay buffer file path
    pretrain_model_save_path: Pre-train policy file path
    """
    # Store all directory names where information is saved
    all_saving_dirs = {}

    # Experiment output
    if args.mode == "experiment":
        model_save_path = saving_dir + "/policy/exp_policy"
        tensorboard_dir = saving_dir + "/output/tensorboard/"
        output_dir = saving_dir + "/output"
        replay_buffer_dir = saving_dir + "/replay_buffer/"  # Directory to hold replay buffer
        heatmap_dir = saving_dir + "/output/heatmap/"
        boxplot_dir = saving_dir + "/output/boxplot/"
        results_saving_dir = saving_dir + "/output/results"
    elif args.mode == "combined" or args.mode == "naive" or args.mode == "position-dependent":
        output_dir = saving_dir + "/output/"
        replay_buffer_dir = saving_dir + "/replay_buffer/"  # Directory to hold replay buffer
        heatmap_dir = output_dir + "heatmap/"
        boxplot_dir = output_dir + "boxplot/"
        model_save_path = "None"
        results_saving_dir = "None"
        tensorboard_dir = "None"
    elif args.mode == "eval":
        output_dir = saving_dir + "/output/"
        heatmap_dir = output_dir + "heatmap/"
        replay_buffer_dir = "None"
        boxplot_dir = "None"
        model_save_path = "None"
        results_saving_dir = "None"
        tensorboard_dir = "None"
    else:
        print("---------- STARTING: ", args.mode, " ---------")
        # Saving directories for model, output, and tensorboard
        model_save_path = saving_dir+"/policy/{}_{}".format(args.mode, "DDPGfD_kinovaGrip")
        output_dir = saving_dir + "/output" # Directory to hold output (plotting, avg. reward data, etc.)
        replay_buffer_dir = saving_dir + "/replay_buffer/" # Directory to hold replay buffer
        tensorboard_dir = output_dir + "/tensorboard/{}".format(args.tensorboardindex)
        boxplot_dir = output_dir + "/boxplot/"
        heatmap_dir = output_dir + "/heatmap/"
        results_saving_dir = output_dir + "/results/"

    # Create directory paths if they do not exist
    if create_dirs is True:
        create_paths([saving_dir, model_save_path, replay_buffer_dir, output_dir, tensorboard_dir, heatmap_dir, boxplot_dir, results_saving_dir])

    all_saving_dirs["saving_dir"] = saving_dir
    all_saving_dirs["model_save_path"] = model_save_path
    all_saving_dirs["output_dir"] = output_dir
    all_saving_dirs["tensorboard_dir"] = tensorboard_dir
    all_saving_dirs["heatmap_dir"] = heatmap_dir
    all_saving_dirs["boxplot_dir"] = boxplot_dir
    all_saving_dirs["results_saving_dir"] = results_saving_dir
    all_saving_dirs["replay_buffer"] = replay_buffer_dir
    all_saving_dirs["expert_replay_file_path"] = expert_replay_file_path
    all_saving_dirs["agent_replay_file_path"] = agent_replay_file_path
    all_saving_dirs["pretrain_model_save_path"] = pretrain_model_save_path
    all_saving_dirs["train_init_coord_file_path"] = env.get_coords_filename()
    all_saving_dirs["eval_init_coord_file_path"] = env.get_coords_filename()
    all_saving_dirs["controller_init_coord_file_path"] = env.get_coords_filename()

    return all_saving_dirs


def get_experiment_info(exp_num):
    """ Get stage and name of current experiment and pre-trained experiment
    exp_num: Experiment number
    """
    # Experiment #: [pretrain_policy_exp #, stage_policy]
    stage0 = "pretrain_policy"  # Expert policy with small cube
    stage1 = {"1": ["0", "sizes"], "2": ["0", "shapes"], "3": ["0", "orientations"]}
    stage2 = {"4": ["1", "sizes_shapes_orientations"], "5": ["2", "shapes_sizes_orientations"], "6": ["3", "orientations_sizes_shapes"]}

    #stage2 = {"4": ["1", "sizes_shapes"], "5": ["1", "sizes_orientations"], "6": ["2", "shapes_orientations"],
    #          "7": ["2", "shapes_sizes"], "8": ["3", "orientations_sizes"], "9": ["3", "orientations_shapes"]}
    #stage3 = {"10": ["4", "sizes_shapes_orientations"], "11": ["5", "sizes_orientations_shapes"],
    #          "12": ["6", "shapes_orientations_sizes"], "13": ["7", "shapes_sizes_orientations"],
    #          "14": ["8", "orientations_sizes_shapes"], "15": ["9", "orientations_shapes_sizes"]}

    print("stage1.keys(): ",stage1.keys())
    print("exp_num: ",exp_num)
    exp_num = str(exp_num)
    if exp_num in stage1.keys():
        prev_exp_stage = "0"
        exp_stage = "1"
        prev_exp_num = "0"
        prev_exp_name = stage0
        exp_name = stage1[exp_num][1]
    elif exp_num in stage2.keys():
        prev_exp_stage = "1"
        exp_stage = "2"
        prev_exp_num = stage2[exp_num][0]
        prev_exp_name = stage1[prev_exp_num][1]
        exp_name = stage2[exp_num][1]
    elif exp_num in stage3.keys():
        prev_exp_stage = "2"
        exp_stage = "3"
        prev_exp_num = stage3[exp_num][0]
        prev_exp_name = stage2[prev_exp_num][1]
        exp_name = stage3[exp_num][1]
    elif exp_num == 16:
        prev_exp_stage = "0"
        exp_stage = "kitchen_sink"
        prev_exp_num = "0"
        prev_exp_name = stage0
        exp_name = "kitchen_sink"
    else:
        print("Invalid experiment option: ", exp_num)
        raise ValueError
    return prev_exp_stage, prev_exp_num, prev_exp_name, exp_stage, exp_name


def get_experiment_file_structure(prev_exp_stage, prev_exp_name, exp_stage, exp_name):
    """ Setup experiment file structure with directories for the policy and plot output
    prev_exp_stage: Prev exp stage
    prev_exp_name: Previous exp name
    exp_stage: Current experiment stage
    exp_name: Current experiment name
    """
    rl_exp_base_dir = "rl_experiments"
    grasp_dir = "/no_grasp"
    if args.with_grasp_reward is True:
        grasp_dir = "/with_grasp"
    stage_dir = "/stage" + exp_stage

    exp_dir = rl_exp_base_dir + grasp_dir + stage_dir + "/" + exp_name
    policy_dir = exp_dir+"/policy"
    replay_dir = exp_dir + "/replay_buffer"
    output_dir = exp_dir+"/output"
    create_paths([exp_dir, policy_dir, replay_dir, output_dir])

    expert_replay_dir = "./expert_replay_data" + grasp_dir + "/combined"
    if not os.path.isdir(expert_replay_dir):
        print("Expert replay buffer experience directory not found!: ", expert_replay_dir)

    prev_exp_dir = rl_exp_base_dir + grasp_dir + "/" + "stage" + prev_exp_stage + "/" + prev_exp_name
    if not os.path.isdir(prev_exp_dir):
        print("Previous experiment directory not found!: ", prev_exp_dir)

    pretrain_replay_dir = prev_exp_dir + "/replay_buffer/"
    if not os.path.isdir(pretrain_replay_dir):
        print("Previous experiment Replay Buffer directory not found!: ", pretrain_replay_dir)

    pretrain_policy_dir = prev_exp_dir + "/replay_buffer/"
    if not os.path.isdir(pretrain_policy_dir):
        print("Previous experiment Policy directory not found!: ", pretrain_policy_dir)

    return expert_replay_dir, prev_exp_dir, exp_dir


def get_exp_input(exp_name, shapes, sizes):
    """ Return the correct shapes, sizes, and orientations based on requested experiment
    exp_name: Experiment name (sizes, shapes, orientations)
    shapes: All shape options
    sizes: All shape sizes
    """
    exp_types = exp_name.split('_')
    exp_shapes = []

    if exp_name == "kitchen_sink":
        exp_types = ["shapes", "sizes", "orientations"]

    # All shapes
    if "shapes" in exp_types and "sizes" in exp_types:
        for size in sizes:
            exp_shapes += [shape + size for shape in shapes]
    elif "shapes" in exp_types:
        exp_shapes += [shape + "S" for shape in shapes]
    elif "sizes" in exp_types:
        exp_shapes += ["Cube" + size for size in sizes]
    else:
        exp_shapes += ["CubeS"]
    # All orientations
    if "orientations" in exp_types:
        exp_orientation = "random"
    else:
        exp_orientation = "normal"

    return exp_shapes, exp_orientation


def generate_output(text, shapes_list, orientations_list, num_success, num_total, all_saving_dirs, frame="local", plot_type="eval"):
    """ Generate heatmaps, boxplots, and output info file """
    # Produce plots

    # Train Heatmap
    if os.path.isdir(all_saving_dirs["heatmap_dir"]) is True:
        print("Generating heatmaps...")
        for orientation in orientations_list:
            for shape in shapes_list:
                plot_output_dir = all_saving_dirs["heatmap_dir"] + orientation + "/" + shape + "/" + frame + "/"
                generate_heatmaps(plot_type=plot_type, shapes_list=shapes_list, orientation=str(orientation), data_dir=plot_output_dir,
                                  saving_dir=plot_output_dir,max_episodes=args.max_episode, saving_freq=args.save_freq)
    else:
        print("Heatmap dir NOT found: ", all_saving_dirs["heatmap_dir"])

    if os.path.isdir(all_saving_dirs["boxplot_dir"]) is True:
        print("Generating boxplots...")
        # Boxplot evaluation reward
        plot_output_dir = all_saving_dirs["boxplot_dir"] + "/"
        generate_reward_boxplots(plot_type=plot_type,orientation=str(requested_orientation), data_dir=plot_output_dir,
                                     saving_dir=plot_output_dir, tot_episodes=args.max_episode, saving_freq=args.save_freq, eval_freq=args.eval_freq)
    else:
        print("Boxplot dir NOT found: ", all_saving_dirs["boxplot_dir"])

    print("Writing to experiment info file...")
    if all_saving_dirs is not None:
        create_info_file(num_success, num_total, all_saving_dirs, text)


def rl_experiment(policy, exp_num, exp_name, prev_exp_dir, requested_shapes, requested_orientation_list, all_saving_dirs, state_idx_arr=np.arange(82)):
    """ Train policy according to RL experiment shape, size, orientation combo + stage """
    # Fill object list using latin square method
    env.Generate_Latin_Square(args.max_episode, "objects.csv", shape_keys=requested_shapes)

    # Get policy name based on what is saved at file
    pol_dir = prev_exp_dir + "/policy/"
    policy_filename_path = glob.glob(pol_dir+"*_actor_optimizer")
    policy_filename = os.path.basename(policy_filename_path[0])
    policy_filename = policy_filename.replace('_actor_optimizer', '')

    # Load Pre-Trained policy
    policy.load(pol_dir+policy_filename)

    # *** Train policy ****
    # TODO: Add in state_idx_arr=state_idx_arr !!!!
    eval_num_success, eval_num_fail = conduct_episodes(policy, controller_type, expert_replay_buffer, replay_buffer, max_episode, expert_prob, "TRAIN", all_saving_dirs, max_num_timesteps, velocities, state_idx_arr=state_idx_arr)
    eval_num_total = eval_num_success + eval_num_fail

    print("Experiment ", exp_num, ", ", exp_name, " policy saved at: ", all_saving_dirs["model_save_path"])

    # Produce plots and output info file
    output_dir = exp_dir + "/output/"
    saving_dir = output_dir

    grasp_text = ""
    if args.with_grasp_reward is True:
        grasp_text = "WITH grasp"
    else:
        grasp_text = "NO grasp"
    exp_text = grasp_text + " Experiment " + str(exp_num) + ": " + exp_name + ", Stage " + str(
        exp_stage) + "\nDate: {}".format(datetime.datetime.now().strftime("%m_%d_%y_%H%M"))
    previous_text = "Previous experiment: " + prev_exp_name
    type_text = "Experiment shapes: " + str(requested_shapes) + "\nExperiment orientation: " + str(
        requested_orientation)
    success_text = "Final Policy Evaluation:\n# Success: " + str(eval_num_success) + "\n# Failures: " + str(
        eval_num_total-eval_num_success) + "\n# Total: " + str(eval_num_total)
    output_text = "Output directory: " + str(exp_dir)

    text = exp_text + "\n" + previous_text + "\n" + type_text + "\n" + success_text + "\n" + output_text

    # Create unique info file
    f = open(all_saving_dirs["output_dir"] + "/"+str(args.mode)+"_info.txt", "w")
    f.write(text)
    f.close()

    # Generate output plots and info file, all saving dirs set to none as it has a unique info file
    generate_output(text=text,  shapes_list=requested_shapes, orientations_list=requested_orientation_list, num_success=eval_num_success, num_total=eval_num_total, all_saving_dirs=all_saving_dirs)

    print("--------------------------------------------------")
    print("Finished Experiment!")
    print(output_text)
    print(previous_text)
    print(type_text)
    print(success_text)
    print(output_text)


def create_info_file(num_success,num_total,all_saving_dirs,extra_text=""):
    """ Create text file containing information about the current training run """
    # INFO FILE
    # HEADER
    # Name: Mode and Policy name
    # Date: Month, Day, Year, Time
    # Saving directory: Main saving directory name
    header_text = "Name: {}\nDate: {}\nSaving Dir: {}".format(all_saving_dirs["saving_dir"], datestr, all_saving_dirs["saving_dir"])

    # INPUT:
    # Policy Initialization: None (Random init) or pre-train filepath
    # Expert Replay Buffer: None or filepath
    # Agent Replay Buffer: None or filepath
    # Object/hand pose initial coordinate text file path
    input_text = "\n\nINPUT:\nPolicy dir: {}\nExpert Replay Buffer: {}\nAgent Replay Buffer: {}\nInitial Object/Hand Pose Coord. File (Controller): {}\nInitial Object/Hand Pose Coord. File (Train): {}\nInitial Object/Hand Pose Coord. File (Evaluation): {}".format(all_saving_dirs["pretrain_model_save_path"], all_saving_dirs["expert_replay_file_path"], all_saving_dirs["agent_replay_file_path"], all_saving_dirs["controller_init_coord_file_path"], all_saving_dirs["train_init_coord_file_path"], all_saving_dirs["eval_init_coord_file_path"])

    # OUTPUT:
    # Policy: None or model_save_path
    # Agent Replay Buffer: None or filepath
    # Output (plotting, results): None or output_dir
    #   Output/ Tensorboard: None or tensorboard_dir
    #   Output/ Heatmap: None or heatmap_dir
    #   Output/ Results: None or results_saving_dir
    policy_output_text = "\n\nOUTPUT:\nPolicy: {}\nAgent Replay Buffer: {}".format(all_saving_dirs["model_save_path"],all_saving_dirs["replay_buffer"])
    plotting_output_text = "\nOutput dir: {}\nOutput/ Tensorboard: {}\nOutput/ Heatmap: {}\nOutput/ Results: {}".format(all_saving_dirs["output_dir"],all_saving_dirs["tensorboard_dir"],all_saving_dirs["heatmap_dir"],all_saving_dirs["results_saving_dir"])
    success_text = "\n---- SUCCESS INFO: ----\n# Success: {}\n# Failures: {}\n# Total: {}".format(str(num_success),str(num_total - num_success),str(num_total)) + "\n"
    # ADDITIONAL TEXT/INFO (text or "") --> TRAINING PARAMETERS are included within extra_text

    print("Writing to info file...")
    f = open(all_saving_dirs["output_dir"] + "/"+str(args.mode)+"_info.txt", "w")

    text = header_text + input_text + policy_output_text + plotting_output_text + success_text + extra_text

    f.write(text)
    f.close()

    print("\n---------------- DONE RUNNING: "+args.mode+" ---------------------")
    print(text)
    print("------------------------------------")


def setup_args(args=None):
    """ Set important variables based on command line arguments OR passed on argument values
    returns: Full set of arguments to be parsed
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="DDPGfD")              # Policy name
    parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0") # OpenAI gym environment name
    parser.add_argument("--seed", default=2, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=100, type=int)     # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=200, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--eval_num", default=100, type=int)          # Number of grasp trials to evaluate over
    parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment for
    parser.add_argument("--start_episode", default=0, type=int)         # Initial episode to evaluate the policy over
    parser.add_argument("--max_episode", default=4000, type=int)       # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")           # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=64, type=int)            # Batch size for both actor and critic - Change to be 64 for batch train, 0 for single ep sample
    parser.add_argument("--discount", default=0.995, type=float)            # Discount factor
    parser.add_argument("--tau", default=0.0005, type=float)                # Target network update rate
    parser.add_argument("--policy_noise", default=0.01, type=float)     # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.05, type=float)       # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates
    parser.add_argument("--tensorboardindex", type=str, default=None)   # Tensorboard log name, found in kinova_gripper_strategy/
    parser.add_argument("--expert_replay_size", default=20000, type=int)    # Number of episode for loading expert trajectories
    parser.add_argument("--saving_dir", type=str, default=None)             # Directory name to save policy within policies/
    parser.add_argument("--shapes", default='CubeS', action='store', type=str) # Requested shapes to use (in format of object keys)
    parser.add_argument("--hand_orientation", action='store', type=str)         # Requested shapes to use (in format of object keys)
    parser.add_argument("--mode", action='store', type=str, default="train")    # Mode to run experiments with: (naive, position-dependent, expert, pre-train, train, rand_train, test, experiment)
    parser.add_argument("--agent_replay_size", default=10000, type=int)         # Maximum size of agent's replay buffer
    parser.add_argument("--expert_prob", default=0.7, type=float)           # Probability of sampling from expert replay buffer (opposed to agent replay buffer)
    parser.add_argument("--with_grasp_reward", type=str, action='store', default="False")  # bool, set True to use Grasp Reward from grasp classifier, otherwise grasp reward is 0
    parser.add_argument("--save_freq", default=200, type=int)  # Frequency to save data at (Ex: every 1000 episodes, save current success/fail coords numpy array to file)
    parser.add_argument("--update_after", default=100, type=int) # Start to update the policy after # episodes have occured
    parser.add_argument("--update_freq", default=1, type=int)   # Update the policy every # of episodes
    parser.add_argument("--update_num", default=100, type=int)  # Number of times to update policy per update step
    parser.add_argument("--exp_num", default=None, type=int)    # RL Paper: experiment number
    parser.add_argument("--render_imgs", type=str, action='store', default="False")   # Set to True to render video images of simulation (caution: will render each episode by default)
    parser.add_argument("--pretrain_policy_path", type=str, action='store', default=None) # Path to the pre-trained policy to be loaded
    parser.add_argument("--agent_replay_buffer_path", type=str, action='store', default=None) # Path to the pre-trained replay buffer to be loaded
    parser.add_argument("--expert_replay_file_path", type=str, action='store', default=None) # Path to the expert replay buffer
    parser.add_argument("--test_policy_path", type=str, action='store', default=None) # Path of the policy to be tested
    parser.add_argument("--test_policy_name", type=str, action='store', default="") # Name of the policy to appear on the rewards plot legend when evaluating that policy
    parser.add_argument("--with_orientation_noise", type=str, action='store', default="False") # Set to true to sample initial hand-object coordinates from with_noise/ dataset
    parser.add_argument("--controller_type", type=str, action='store', default=None) # Determine the type of controller to use for evaluation (policy OR naive, expert, position-dependent)
    parser.add_argument("--regions_of_interest", type=str, action='store', default=None) # Determine the region of interest to evaluate over (Ex: extreme_left, extreme_right, all_regions)
    parser.add_argument("--input_variations", type=str, action='store', default=None)  # Determine the input variations to evaluate over (Ex: Baseline, Baseline_HOV, Shapes_HOV, all_variations)
    parser.add_argument("--replay_buffer_sample_size", type=str, action='store', default=100) # Number of entries to load from the end of the replay buffer
    parser.add_argument("--sampling_decay_rate", type=float, action='store', default=0.2) # Rate of decay of the proportion of replay buffer experience that is samples from the expert within training

    # state dim related arguments
    parser.add_argument("--state_range", default='all',
                        type=str)  # string - from ('all', 'nigel_rangefinder', 'nigel_norangefinder', 'all_real', 'adam_sim2real')

    args = parser.parse_args()
    return args


def test_policy_models_match(current_model, compare_model=None, compare_model_filepath=None):
    """ Check that the input models match. This can check that we have loaded in the trained model correctly. """

    # TODO, STATE DIM STUFF: Does this need to be changed to match policy
    # NOTE: THIS DOESN'T GET USED ANYWHERE ATM. don't touch it LOLOLOL

    # Set dimensions for state and action spaces - policy initialization
    state_dim = 82  # State dimension dependent on the length of the state space
    action_dim = 3
    max_action = 0.8
    n = 5   # n step look ahead for the policy
    max_q_value = 50 # Should match the maximum reward value

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "n": n,
        "discount": 0.995,
        "tau": 0.0005,
        "batch_size": 64,
        "expert_sampling_proportion": 0.7
    }

    # Get the comparison policy from a saved location
    if compare_model_filepath is not None:
        ''' Initialize policy '''
        compare_model = DDPGfD.DDPGfD(**kwargs)

        print("In test_policy_models_match! loading saved model: ", compare_model_filepath)
        compare_model.load(compare_model_filepath)

    current_networks = {"actor": current_model.actor, "critic": current_model.critic}
    saved_networks = {"actor": compare_model.actor, "critic": compare_model.critic}

    # Check the current and saved models (torch nn) match
    for (current_model_key, current_model_value), (compare_model_key, compare_model_value) in zip(current_networks.items(), saved_networks.items()):
        print("Models being evaluated: {}, {}".format(current_model_key, compare_model_key))
        # Check that the models match
        models_differ = 0
        for key_item_1, key_item_2 in zip(current_model_value.state_dict().items(), compare_model_value.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                    return False
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly!')
            return True

def create_input_variation_reward_plot(policies, start_episode, eval_freq, max_episode):
    """ Plot the reward for each policy for each input variation (Baseline, Baseline + HOV, Shapes + HOV)"""
    policy_colors = {"Baseline": "#808080", "Baseline_HOV": "black", "Sizes_HOV": "blue", "Shapes_HOV": "red", "Orientations_HOV": "#ffd900"}
    # variation_input_policies format: For the current variation type, we get {policy_name: rewards}

    if eval_freq == 0:
        policy_eval_points = np.array([0])
    else:
        num_policies = int((max_episode-start_episode) / eval_freq) + 1
        policy_eval_points = np.linspace(start=start_episode, stop=max_episode, num=num_policies, dtype=int)

    for policy_name, policy_filepath in policies.items():
        # Read reward data
        with open(policy_filepath + "/" + str(test_policy_name) + "_policy_rewards.txt") as f:
            data = f.read()

        policy_rewards = json.loads(data)
        # AFTER we have gone through each evaluation point, record the current policy's rewards for each variation
        variation_rewards_per_policy[policy_name] = policy_rewards

    # Create a reward plot for each variation input type over evaluation points in training
    for variation_type in variations:
        variation_input_name = variation_type["variation_name"]
        variation_input_policies = {}

        # Get the reward data from each policy for the current variation_input_name
        for policy_name in policies:
            policy_rewards_dict = variation_rewards_per_policy[policy_name]
            # For the CURRENT VARIATION: {"Baseline": [reward, reward, ..]
            variation_input_policies[policy_name] = policy_rewards_dict[variation_input_name]

        # Save reward data to file
        dict_file = open(saving_dir + "/" +str(variation_input_name) + "_policy_rewards.csv", "w", newline='')
        keys = variation_input_policies.keys()
        dict_writer = csv.DictWriter(dict_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows([variation_input_policies])
        dict_file.close()

        # Create a reward plot for variation_input_name for evaluation of the policy over time
        reward_plot(policy_eval_points, variation_input_policies, variation_input_name, policy_colors, start_episode, eval_freq, max_episode, saving_dir)

def state_dim_setup(state_dim_option):
    """
    Returns an array of indices that can be used on a full observation to only grab relevant state dimensions
    Input: The argument of which state_range option to use.
    Output: Numpy array of indices
    """

    # Setup state dimensional parts here
    '''
    Local obs, all in local coordinates (from the center of the palm)
    (18,) Finger Pos                                        0-17: (0: x, 1: y, 2: z) "f1_prox", (3-5) "f2_prox", (6-8) "f3_prox", (9-11) "f1_dist", (12-14) "f2_dist", (15-17) "f3_dist"
    (3,) Wrist Pos                                          18-20 (18: x, 19: y, 20: z)
    (3,) Obj Pos                                            21-23 (21: x, 22: y, 23: z)
    (9,) Joint States                                       24-32
    (3,) Obj Size                                           33-35
    (12,) Finger Object Distance                            36-47

    36) "f1_prox"
    37) "f1_prox_1"
    38) "f2_prox"
    39) "f2_prox_1"
    40) "f3_prox"
    41) "f3_prox_1"
    42) "f1_dist"
    43) "f1_dist_1"
    44) "f2_dist"
    45) "f2_dist_1"
    46) "f3_dist"
    47) "f3_dist_1"

    Note: NONE vs "_1" meaning: On each finger there are two red dots. The "_1" is the ones closer to the center


    (2,) X and Z angle                                      48-49
    (17,) Rangefinder data                                  50-66
    (3,) Gravity vector in local coordinates                67-69
    (3,) Object location based on rangefinder data          70-72
    (1,) Ratio of the area of the side of the shape to the open portion of the side of the hand    73
    (1,) Ratio of the area of the top of the shape to the open portion of the top of the hand    74
    (6, ) Finger dot product  75) "f1_prox", 76) "f2_prox", 77) "f3_prox", 78) "f1_dist", 79) "f2_dist", 80) "f3_dist"  75-80
    (1, ) Dot product (wrist) 81
    '''

    finger_pos_idx = np.arange(0, 18)
    f1_prox_pos_idx = np.array([0, 1, 2])
    f2_prox_pos_idx = np.array([3, 4, 5])
    f1_dist_pos_idx = np.array([9, 10, 11])
    f2_dist_pos_idx = np.array([12, 13, 14])
    last_6_joint_states_idx = np.arange(27, 33)
    f1_f2_joint_angles_idx = np.array([27, 28, 30,
                                       31])  # TODO: figure out the right angles here... => in order: f1 proximal angle joint, f2 proximal angle joint, f1 distal angle joint, f2 distal angle joint.
    # note: these are also all in radians in simulator...

    wrist_pos_idx = np.arange(18, 21)
    obj_pos_idx = np.arange(21, 24)
    joint_states_idx = np.arange(24, 33)
    obj_size_idx = np.arange(33, 36)
    finger_obj_dist_idx = np.arange(36, 48)

    finger_obj_dist_f1_prox_1 = np.array([37])
    finger_obj_dist_f2_prox_1 = np.array([39])
    finger_obj_dist_f1_dist_1 = np.array([43])
    finger_obj_dist_f2_dist_1 = np.array([45])

    x_z_angle_idx = np.arange(48, 50)
    rangefinder_data_idx = np.arange(50, 67)
    gravity_vector_in_local_coords = np.arange(67, 70)
    object_location_rangefinder = np.arange(70, 73)
    ratio_sideshape_sidehand = np.array([73])
    ratio_topshape_tophand = np.array([74])
    f1_prox_idx = np.array([75])
    f2_prox_idx = np.array([76])
    f3_prox_idx = np.array([77])
    f1_dist_idx = np.array([78])
    f2_dist_idx = np.array([79])
    f3_dist_idx = np.array([80])
    dot_prod_wrist = np.array([81])

    # create mappings for state dimension mapping
    state_dim_idx_arr_dict = {
        'all': np.arange(82),
        'nigel_rangefinder': np.concatenate((obj_pos_idx, rangefinder_data_idx, obj_size_idx), axis=0),
        'nigel_norangefinder': np.concatenate((obj_pos_idx, finger_obj_dist_idx, obj_size_idx), axis=0),
        'all_real': np.concatenate((f1_prox_pos_idx, f2_prox_pos_idx, f1_dist_pos_idx, f2_dist_pos_idx, obj_pos_idx,
                                    last_6_joint_states_idx, obj_size_idx, finger_obj_dist_idx)),
        #  wrist 3 + finger pos 12 + obj size 3 + last joint states 6 + obj pos 3 + finger obj dist 4
        'adam_sim2real': np.concatenate((f1_dist_pos_idx, f1_prox_pos_idx, f2_dist_pos_idx, f2_prox_pos_idx,
                                         wrist_pos_idx, obj_pos_idx, last_6_joint_states_idx, obj_size_idx,
                                         finger_obj_dist_f1_dist_1, finger_obj_dist_f1_prox_1,
                                         finger_obj_dist_f2_dist_1, finger_obj_dist_f2_prox_1)), # finger positions (12), wrist pos (3), obj pos (3), last 6 joint states (6), obj size (3), finger obj dist (4)
        # this one is based on sim2real
        'adam_sim2real_v02': np.concatenate((f1_prox_pos_idx, f1_dist_pos_idx, f2_prox_pos_idx, f2_dist_pos_idx,
                                             obj_pos_idx, f1_f2_joint_angles_idx, obj_size_idx,
                                             finger_obj_dist_f1_prox_1, finger_obj_dist_f1_dist_1,
                                             finger_obj_dist_f2_prox_1, finger_obj_dist_f2_dist_1))
        # this one removes wrist position and third finger joints.
    }

    assert state_dim_option in state_dim_idx_arr_dict.keys()

    res_state_idx_arr = state_dim_idx_arr_dict[state_dim_option]

    return res_state_idx_arr


if __name__ == "__main__":
    # Set up environment based on command-line arguments or passed in arguments
    args = setup_args()
    """ Setup the environment, state, and action space """

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: "+file_name)
    print("---------------------------------------")

    # Date string to stay consistent over file naming
    datestr = "_{}".format(datetime.datetime.now().strftime("%m_%d_%y_%H%M"))

    # Make initial environment
    env = gym.make(args.env_name)

    # Set seeds for randomization
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TODO: state dimensional stuff

    state_idx_arr = state_dim_setup(args.state_range)
    modified_state_dim = len(state_idx_arr)
    print("============= MODIFIED STATE DIM: ", modified_state_dim, " =============================")

    # Set dimensions for state and action spaces - policy initialization
    state_dim = 82  # State dimension dependent on the length of the state space
    action_dim = 3 #env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_action_trained = env.action_space.high  # a vector of max actions
    n = 5   # n step look ahead for the policy
    max_q_value = 50  # Should match the maximum reward value
#     velocities = {"constant_velocity": 2, "min_velocity": 0, "max_velocity": 3, "finger_lift_velocity": 1, "wrist_lift_velocity": 1}
    velocities = {"constant_velocity": 1.5, "min_velocity": 0, "max_velocity": 1.5, "finger_lift_velocity": 1, "wrist_lift_velocity": 1}

    ''' Set values from command line arguments '''
    requested_shapes = args.shapes                   # Sets list of desired objects for experiment
    requested_shapes = requested_shapes.split(',')
    requested_orientation = args.hand_orientation   # Set the desired hand orientation (normal or random)
    expert_replay_size = args.expert_replay_size    # Number of expert episodes for expert the replay buffer
    agent_replay_size = args.agent_replay_size      # Maximum number of episodes to be stored in agent replay buffer
    replay_buffer_sample_size = args.replay_buffer_sample_size # Number of entries to load the replay buffer with from the saved replay buffer
                                                               # (sampled from the end of the buffer to get the most recent experience)
    if replay_buffer_sample_size == "None":
        replay_buffer_sample_size = None
    sampling_decay_rate = args.sampling_decay_rate
    max_num_timesteps = 60     # Maximum number of time steps within an episode

    # If experiment number is selected, set mode to experiment (in case the mode has been set to train by default)
    if args.exp_num is not None:
        args.mode = "experiment"

    # Set requested_orientation_list for directory creation, plotting and reference
    if requested_orientation == "random":
        requested_orientation_list = ["normal", "rotated", "top"]
    else:
        requested_orientation_list = [requested_orientation]

    # Fill pre-training object list using latin square method
    env.Generate_Latin_Square(args.max_episode,"objects.csv", shape_keys=requested_shapes)

    kwargs = {
        "state_dim": modified_state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "n": n,
        "discount": args.discount,
        "tau": args.tau,
        "batch_size": args.batch_size,
        "expert_sampling_proportion": args.expert_prob
    }

    ''' Initialize policy '''
    if args.policy_name == "DDPGfD":
        policy = DDPGfD.DDPGfD(**kwargs)
    else:
        print("No such policy")
        raise ValueError

    # Determine the type of controller (policy, naive, position-dependent)
    if args.controller_type is None:
        print("You did determine the controller type! --controller_type by default is None")
        raise ValueError
    else:
        controller_type = args.controller_type

    # Set grasp reward based on command line input
    if args.with_grasp_reward == "True" or args.with_grasp_reward == "true":
        args.with_grasp_reward = True
        with_grasp_str = "with_grasp"
    elif args.with_grasp_reward == "False" or args.with_grasp_reward == "false":
        args.with_grasp_reward = False
        with_grasp_str = "no_grasp"
    else:
        print("with_grasp_reward must be True or False")
        raise ValueError

    # Set with orientation noise based on command line input
    if args.with_orientation_noise == "True" or args.with_orientation_noise == "true":
        with_orientation_noise = True
        noise_str = "with_noise"
    elif args.with_orientation_noise == "False" or args.with_orientation_noise == "false":
        with_orientation_noise = False
        noise_str = "no_noise"
    else:
        print("with_orientation_noise must be True or False")
        raise ValueError

    if args.render_imgs == "True" or args.render_imgs == "true":
        args.render_imgs = True
    else:
        args.render_imgs = False

    experiment_dir = "./experiments/"
    experiment_mode_dir = experiment_dir + str(args.mode) + "/"
    create_paths(["./experiments/", experiment_mode_dir])
    saving_dir = args.saving_dir
    if saving_dir is None:
        saving_dir = "%s_%s" % (args.policy_name, args.mode) + datestr

    if args.mode == "naive" or args.mode == "position-dependent" or args.mode == "combined":
        exp_grasp_noise_dir = experiment_mode_dir + noise_str + "/" + with_grasp_str + "/"
        create_paths([exp_grasp_noise_dir])
        saving_dir = exp_grasp_noise_dir + "/{}/".format(str(requested_shapes[0])) + requested_orientation
        if os.path.isdir(saving_dir):
            saving_dir = saving_dir + datestr
    else:
        saving_dir = experiment_mode_dir + "{}/".format(saving_dir)
        if os.path.isdir(saving_dir):
            saving_dir = saving_dir + datestr
    create_paths([saving_dir])

    if args.tensorboardindex is None and controller_type == "policy":
        args.tensorboardindex = "%s_%s" % (args.policy_name, args.mode)
        args.tensorboardindex = args.tensorboardindex[:30]  # Keep the tensorboard name at a max size of 30 characters

    max_episode = args.max_episode
    expert_prob = args.expert_prob
    eval_freq = args.eval_freq
    eval_num = args.eval_num
    start_episode = args.start_episode

    # Set the variation input type for evaluation
    if args.input_variations == "None" or args.input_variations is None:
        input_variations = None
    elif args.input_variations == "all_variations":
        input_variations = "all_variations"
    else:
        input_variations = args.input_variations.split(',')

    # Set the regions of interest for evaluation
    if args.regions_of_interest == "None" or args.regions_of_interest is None:
        regions_of_interest = None
    elif args.regions_of_interest == "all_regions":
        regions_of_interest = "all_regions"
    else:
        regions_of_interest = args.regions_of_interest.split(',')

    # Print variables set based on command line input
    param_text = ""
    if args.mode == "experiment":
        param_text += "Grasp Reward: "+ str(args.with_grasp_reward) + "\n"
        param_text += "Running EXPERIMENT: "+str(args.exp_num) + "\n"
    else:
        param_text += "Saving dir: "+ str(saving_dir) + "\n"
        param_text += "Seed: " + str(args.seed) + "\n"
        param_text += "Tensorboard index: "+str(args.tensorboardindex) + "\n"
        param_text += "Policy: "+ str(args.policy_name) + "\n"
        param_text += "Requested_shapes: "+str(requested_shapes) + "\n"
        param_text += "Requested Hand orientation: "+ str(requested_orientation) + "\n"
        param_text += "With Orientation Noise: " + str(args.with_orientation_noise) + "\n"
        param_text += "Batch Size: "+ str(args.batch_size) + "\n"
        param_text += "Expert Sampling Probability: "+ str(expert_prob) + "\n"
        param_text += "Grasp Reward: "+ str(args.with_grasp_reward) + "\n"
        param_text += "Regions of Interest: " + str(regions_of_interest) + "\n"
        param_text += "Save frequency: "+ str(args.save_freq) + "\n"
        param_text += "Evaluation frequency: " + str(args.eval_freq) + "\n"
        param_text += "Number of Evaluation episodes: " + str(args.eval_num) + "\n"
        param_text += "Policy update after: "+ str(args.update_after) + "\n"
        param_text += "Policy update frequency: "+ str(args.update_freq) + "\n"
        param_text += "Policy update Amount: "+ str(args.update_num) + "\n"
        if args.mode != "position-dependent" and args.mode != "naive" and args.mode != "combined":
            param_text += "Generating " + str(max_episode) + " episodes!"
        print("\n----------------- SELECTED MODE: ", str(args.mode), "-------------------------")
        print("PARAMETERS: \n")
        print(param_text)
        print("\n-------------------------------------------------")

    ''' Select replay buffer type, previous buffer types
    # Replay buffer that can use multiple n-steps
    #replay_buffer = utils.ReplayBuffer_VarStepsEpisode(state_dim, action_dim, expert_replay_size)
    #replay_buffer = utils.ReplayBuffer_NStep(state_dim, action_dim, expert_replay_size)

    # Replay buffer that samples only one episode
    #replay_buffer = utils.ReplayBuffer_episode(state_dim, action_dim, env._max_episode_steps, args.expert_replay_size, args.max_episode)
    #replay_buffer = GenerateExpertPID_JointVel(args.expert_replay_size, replay_buffer)
    '''

    ## Expert Replay Buffer ###
    if expert_prob > 0 and args.expert_replay_file_path is None:
        expert_replay_file_path = experiment_dir + "position-dependent/" + noise_str + "/" + with_grasp_str + "/"
    elif expert_prob == 0 or args.expert_replay_file_path is None:
        expert_replay_file_path = None
    elif args.expert_replay_file_path is not None:
        expert_replay_file_path = args.expert_replay_file_path
    else:
        expert_replay_file_path = experiment_dir + "naive/" + noise_str + "/" + with_grasp_str + "/"

    ## Agent Replay Buffer ##
    agent_replay_file_path = args.agent_replay_buffer_path # FILL WITH AGENT REPLAY FROM PRETRAINING

    ## Pre-trained Policy ##
    # Default pre-trained policy file path
    pretrain_model_save_path = args.pretrain_policy_path
    if args.test_policy_path == "None":
        test_policy_path = None
    else:
        test_policy_path = args.test_policy_path
    test_policy_name = args.test_policy_name

    # Initialize timer to analyze run times
    total_time = Timer()
    total_time.start()

    # Determine replay buffer/policy function calls based on mode (expert, pre-train, train, evaluate)
    # Generate expert data based on Naive controller only
    if args.mode == "naive" or args.mode == "position-dependent" or args.mode == "combined":
        print("MODE: " + args.mode)
        # Create directories where information will be saved
        all_saving_dirs = setup_directories(env, saving_dir, "None", "None", "None",create_dirs=True)

        # The controller only fills the replay_buffer, it does not access any previous expert replay buffers
        expert_buffers = None

        # Initialize expert replay buffer, then generate expert pid data to fill it
        replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)

        num_success, num_fail = conduct_episodes(policy, controller_type, expert_buffers, replay_buffer, max_episode, expert_prob, args.mode, all_saving_dirs, max_num_timesteps, velocities, state_idx_arr=state_idx_arr)

        print(args.mode + " saving directory: ", all_saving_dirs["saving_dir"])
        print(args.mode + " replay buffer file path: ",all_saving_dirs["replay_buffer"])

        generate_output(text="\nPARAMS: \n"+param_text, shapes_list=requested_shapes, orientations_list=requested_orientation_list, num_success=num_success, num_total=num_success+num_fail, all_saving_dirs=all_saving_dirs,plot_type=None)

    # Pre-train policy using expert data, save pre-trained policy for use in training
    elif args.mode == "pre-train":
        print("MODE: Pre-train")
        print("Expert replay Buffer: ", expert_replay_file_path)
        agent_replay_file_path = None
        print("Agent replay Buffer: ", agent_replay_file_path)

        # Initialize Queue Replay Buffer: replay buffer manages its size like a queue, popping off the oldest episodes
        replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, agent_replay_size)

        # Determine the expert replay buffer(s) to be used based on the requested shapes
        if expert_replay_file_path is None:
            expert_buffers = None
        else:
            expert_buffers = {}
            for shape_to_load in requested_shapes:
                for orientation_to_load in requested_orientation_list:
                    expert_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
                    shape_replay_file_path = expert_replay_file_path + shape_to_load + "/" + str(orientation_to_load) + "/replay_buffer/"
                    # Load expert data from saved expert pid controller replay buffer
                    print("Loading expert replay buffer: ", shape_replay_file_path)
                    replay_text = expert_buffer.store_saved_data_into_replay(shape_replay_file_path)
                    expert_buffers[shape_to_load] = copy.deepcopy(expert_buffer)

        # Create directories where information will be saved
        all_saving_dirs = setup_directories(env, saving_dir, expert_replay_file_path, agent_replay_file_path, pretrain_model_save_path)

        # Initialize timer to analyze run times
        train_time = Timer()
        train_time.start()
        # Pre-train policy based on expert data
        policy.sampling_decay_rate = 0
        eval_num_success, eval_num_fail = conduct_episodes(policy, controller_type, expert_buffers, replay_buffer, max_episode, expert_prob, "PRE-TRAIN", all_saving_dirs, max_num_timesteps, velocities, state_idx_arr=state_idx_arr)
        eval_num_total = eval_num_success + eval_num_fail

        train_time_text = "\nTRAIN time: \n" + train_time.stop()
        print(train_time_text)
        print("\nTrain complete! Now saving...")
        agent_replay_file_path = all_saving_dirs["replay_buffer"] + "/"

        # Create plots and info file
        generate_output(text="\nPARAMS: \n"+param_text+train_time_text+"\n"+replay_text,  shapes_list=requested_shapes, orientations_list=requested_orientation_list, num_success=eval_num_success, num_total=eval_num_total, all_saving_dirs=all_saving_dirs)

    # Train policy starting with pre-trained policy and sampling from experience
    elif args.mode == "train":
        print("MODE: Train (w/ pre-trained policy")
        print("Expert replay Buffer: ", expert_replay_file_path)
        print("Agent replay Buffer: ", agent_replay_file_path)
        print("Pre-trained Policy: ", pretrain_model_save_path)

        # Initialize Queue Replay Buffer: replay buffer manages its size like a queue, popping off the oldest episodes
        replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, agent_replay_size)
        if agent_replay_file_path is not None:
            # Fill experience from previous stage into replay buffer
            replay_buffer.store_saved_data_into_replay(agent_replay_file_path,sample_size=replay_buffer_sample_size)
        else:
            print("Using an empty agent replay buffer!!")

        # Determine the expert replay buffer(s) to be used based on the requested shapes
        if expert_replay_file_path is None:
            expert_buffers = None
        else:
            expert_buffers = {}
            for shape_to_load in requested_shapes:
                for orientation_to_load in requested_orientation_list:
                    expert_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
                    shape_replay_file_path = expert_replay_file_path + shape_to_load + "/" + str(orientation_to_load) + "/replay_buffer/"
                    # Load expert data from saved expert pid controller replay buffer
                    print("Loading expert replay buffer: ", shape_replay_file_path)
                    replay_text = expert_buffer.store_saved_data_into_replay(shape_replay_file_path)
                    expert_buffers[shape_to_load] = copy.deepcopy(expert_buffer)

        # Load Pre-Trained policy
        if pretrain_model_save_path is None:
            print("pretrain_model_save_path is None!! Using random init policy...")
            pretrain_model_save_path = "None (Using random init policy)"
        else:
            policy.load(pretrain_model_save_path)

        # Create directories where information will be saved
        all_saving_dirs = setup_directories(env, saving_dir, expert_replay_file_path, agent_replay_file_path, pretrain_model_save_path)

        # Train the policy and save it
        # Initialize timer to analyze run times
        train_time = Timer()
        train_time.start()
        policy.sampling_decay_rate = sampling_decay_rate
        eval_num_success, eval_num_fail = conduct_episodes(policy, controller_type, expert_buffers, replay_buffer, max_episode, expert_prob, "TRAIN", all_saving_dirs, max_num_timesteps, velocities, state_idx_arr=state_idx_arr)
        eval_num_total = eval_num_success + eval_num_fail

        train_time_text = "\nTRAIN time: \n" + train_time.stop()
        print(train_time_text)
        print("\nTrain complete!")

        # Create plots and info file
        generate_output(text="\nPARAMS: \n"+param_text+train_time_text,  shapes_list=requested_shapes, orientations_list=requested_orientation_list, num_success=eval_num_success, num_total=eval_num_total, all_saving_dirs=all_saving_dirs)

    # Test policy over certain number of episodes -- In Progress
    elif args.mode == "eval":
        print("MODE: Evaluate")
        print("Policy: ", test_policy_path)
        print("Policy Name: ", test_policy_name)
        print("Start Episode: {}\nMax episode: {}\nEvaluation Frequency: {}\nNumber of Evaluation Episodes: {}\nRegions of Interest: {}\n".format(start_episode, max_episode,eval_freq,eval_num,regions_of_interest))

        # Important command line args: --max_episode, --start_episode, --eval_freq, --eval_num, --regions_of_interest, --input_variations, --test_policy_path, --test_policy_name
        # We get the policy from the test_policy_path
        # We SAVE the evaluation output in the area of the saving_dir

        # Determine whether we're evaluating over the policies or comparing performance of a certain controller
        controller_type = str(args.controller_type)

        # Evaluation input variations
        Baseline = {"variation_name": "Baseline", "requested_shapes": ["CubeM"], "requested_orientation": "normal", "with_orientation_noise": False}
        Baseline_HOV = {"variation_name": "Baseline_HOV", "requested_shapes": ["CubeM"], "requested_orientation": "normal", "with_orientation_noise": True}
        Sizes_HOV = {"variation_name": "Sizes_HOV", "requested_shapes": ["CubeS","CubeM","CubeB"], "requested_orientation": "normal", "with_orientation_noise": True}
        Shapes_HOV = {"variation_name": "Shapes_HOV", "requested_shapes": ["CubeM", "CylinderM", "Vase1M"], "requested_orientation": "normal", "with_orientation_noise": True}
        Orientations_HOV = {"variation_name": "Orientations_HOV", "requested_shapes": ["CubeM"], "requested_orientation": "random", "with_orientation_noise": True}

        # Contains all input variation types
        variations_dict = {"Baseline": Baseline, "Baseline_HOV": Baseline_HOV, "Sizes_HOV": Sizes_HOV, "Shapes_HOV": Shapes_HOV, "Orientations_HOV": Orientations_HOV}

        if input_variations is None or input_variations == "all_variations":
            all_variations = variations_dict.keys()
            input_variations = list(all_variations)

        variations = [variations_dict[var_type] for var_type in input_variations]
        print("Input variations: {}\n".format(variations))

        if eval_freq == 0:
            policy_eval_points = np.array([0])
        else:
            num_policies = int((max_episode-start_episode) / eval_freq) + 1
            policy_eval_points = np.linspace(start=start_episode, stop=max_episode, num=num_policies, dtype=int)

        # All rewards (over each evaluation point) for each policy per variation type
        variation_rewards_per_policy = {}

        # For each evaluation point, append the avg. reward from evaluating the policy
        # PER POLICY This will contain the current policy's list of vg. rewards over each evaluation point
        policy_rewards = {}
        policy_rewards["Baseline"] = [{"orientation_shape": ["normal","CubeM"], "rewards": []}]
        policy_rewards["Baseline_HOV"] = [{"orientation_shape": ["normal","CubeM"], "rewards": []}]
        policy_rewards["Sizes_HOV"] = [{"orientation_shape": ["normal","CubeS"], "rewards": []},{"orientation_shape": ["normal","CubeM"], "rewards": []},{"orientation_shape": ["normal","CubeB"], "rewards": []}]
        policy_rewards["Shapes_HOV"] = [{"orientation_shape": ["normal","CubeM"], "rewards": []}, {"orientation_shape": ["normal","CylinderM"], "rewards": []}, {"orientation_shape": ["normal","Vase1M"], "rewards": []}]
        policy_rewards["Orientations_HOV"] = [{"orientation_shape": ["normal","CubeM"], "rewards": []}, {"orientation_shape": ["rotated","CubeM"], "rewards": []}, {"orientation_shape": ["top","CubeM"], "rewards": []}]

        for idx in range(len(policy_eval_points)):

            if controller_type == "policy":
                # Load policy from the evaluation point
                ep_num = policy_eval_points[idx]

                # Do not load a policy, evaluate a random policy
                if test_policy_path is None:
                    eval_point_saving_dir = saving_dir
                    print("Using a random policy!!")
                else:
                    # Load a policy
                    if eval_freq == 0:
                        eval_point_str = ""
                    else:
                        eval_point_str = "/policy_" + str(ep_num) + "/"

                    eval_point_policy_path = test_policy_path + eval_point_str
                    eval_point_saving_dir = saving_dir

                    print("Loading policy: ",eval_point_policy_path)
                    policy.load(eval_point_policy_path)

            else:
                ep_num = ""
                eval_point_saving_dir = saving_dir
                print("Using controller_type: ", controller_type)

            for variation_type in variations:
                variation_name = variation_type["variation_name"]
                variation_saving_dir = saving_dir + "/" + variation_type["variation_name"]
                variation_saving_dirs = setup_directories(env, variation_saving_dir, expert_replay_file_path, agent_replay_file_path, pretrain_model_save_path)

                print("Now evaluating: ", variation_type.items())
                print("eval_point_saving_dir: ",eval_point_saving_dir)
                print("variation_saving_dirs[output_dir]: ", variation_saving_dirs["output_dir"])

                for orientation in [variation_type["requested_orientation"]]:
                    for shape in variation_type["requested_shapes"]:
                        print("\n**Evaluating the policy: {}\nVariation Input: {}\nOrientation: {}\nShape: {}".format(test_policy_name,variation_name,orientation,shape))
                        # Evaluate policy over certain number of episodes
                        eval_ret = eval_policy(policy, args.env_name, args.seed, requested_shapes=[shape], requested_orientation=orientation,
                                               velocities=velocities, controller_type=controller_type,  max_num_timesteps=max_num_timesteps, all_saving_dirs=variation_saving_dirs, eval_episodes=eval_num, render_imgs=args.render_imgs, with_noise=variation_type["with_orientation_noise"], state_idx_arr=state_idx_arr)

                        # Append the average reward from evaluation to the specific variation type avg. reward list
                        for rewards_combo_dict in policy_rewards[variation_name]:
                            if rewards_combo_dict["orientation_shape"][0] == orientation and rewards_combo_dict["orientation_shape"][1] == shape:
                                rewards_combo_dict["rewards"].append(eval_ret["avg_reward"])

                        # All_hand_object_coords is a dictionary containing each hand and object coord. used within evaluation
                        all_hand_object_coords = eval_ret["all_hand_object_coords"]

                        # Sorts coordinates by success/failure per hand orientation
                        sort_and_save_heatmap_coords(all_hand_object_coords, [shape], [orientation], episode_num=None, saving_dir=variation_saving_dirs["heatmap_dir"])

                        # Save the hand and object coordinates -- within the current policy's variation folder (Ex: Policy_0/Baseline/)
                        dict_file = open(variation_saving_dirs["output_dir"]+"/all_hand_object_coords.csv", "w", newline='')
                        keys = all_hand_object_coords[0].keys()
                        dict_writer = csv.DictWriter(dict_file, keys)
                        dict_writer.writeheader()
                        dict_writer.writerows(all_hand_object_coords)
                        dict_file.close()

                        # Generate heatmap plots based on each orientation type used in evaluation (and per coordinate frame - Local, Global, Local-->Global)
                        generate_heatmaps_by_orientation_frame(variation_type, shape, orientation, ["local"], ep_num, all_hand_object_coords, variation_saving_dirs["heatmap_dir"])

                        ## RENDER AND PLOT COORDINATES BY REGION
                        if regions_of_interest is not None:
                            evaluate_coords_by_region(policy, all_hand_object_coords, variation_type, variation_saving_dirs, velocities=velocities, regions_of_interest=regions_of_interest, controller_type=controller_type, state_idx_arr=state_idx_arr)

                # Once the evaluation of a certain variation is complete, save the reward to a text file
                # Save all reward data to file
                variation_reward_dict_save_file = variation_saving_dir + "/Var_Input_" + str(variation_name) + "_rewards.txt"
                with open(variation_reward_dict_save_file, 'w') as convert_file:
                    convert_file.write(json.dumps(policy_rewards[variation_name]))

        # Save all reward data to file
        rewards_dict_save_file = saving_dir + "/" + str(test_policy_name) + "_policy_rewards.txt"
        with open(rewards_dict_save_file, 'w') as convert_file:
            convert_file.write(json.dumps(policy_rewards))

        policies = {}
        policies[test_policy_name] = saving_dir

        create_input_variation_reward_plot(policies, start_episode, eval_freq, max_episode)

    # Experiments for RL paper
    elif args.mode == "experiment":
        # Initialize all shape and size options
        all_shapes = env.get_all_objects()
        test_shapes = ["Vase1", "RBowl"]
        test_sizes = ["M"]
        all_sizes = ["S", "M", "B"]
        exp_mode = "train"  # Manually setting mode right now

        shapes_list = []    # Holds all possible shapes
        for shape_key in all_shapes.keys():
            if shape_key[-1] == "S":
                shapes_list.append(shape_key[:-1])

        train_sizes = ["S", "B"]
        train_shapes = ["Cube", "Cylinder", "Cube45", "Vase2", "Bottle", "Bowl", "TBottle"]

        if exp_mode == "test":
            shapes = test_shapes
            sizes = test_sizes
        else:
            shapes = train_shapes
            sizes = train_sizes

        # Get experiment and stage number from dictionary
        exp_num = args.exp_num
        prev_exp_stage, prev_exp_num, prev_exp_name, exp_stage, exp_name = get_experiment_info(exp_num)

        # Setup directories for experiment output
        expert_replay_file_path, prev_exp_dir, exp_dir = get_experiment_file_structure(prev_exp_stage, prev_exp_name, exp_stage, exp_name)

        # Get experiment-specific shapes list and orientation combo
        requested_shapes, requested_orientation = get_exp_input(exp_name, shapes, sizes)
        if requested_orientation == "random":
            requested_orientation_list = ["normal", "rotated", "top"]
        else:
            requested_orientation_list = ["normal"]

        print("\n------------------------------------------------------")
        print("EXPERIMENT: ", str(args.exp_num), ", ", with_grasp_str)
        print("Stage: ", exp_stage, ", Exp: ", args.exp_num)
        print("Exp. Name: ", exp_name)
        print("Experiment shapes: ", requested_shapes)
        print("Experiment orientation: ", requested_orientation)
        print("Output directory: ", exp_dir)

        print("\nPrev. Exp Stage: ", prev_exp_stage, "Exp: ", prev_exp_num)
        print("Prev. Exp Name: ", prev_exp_name)
        print("Prev. Stage directory: ", prev_exp_dir)
        print("------------------------------------------------------")

        # Initialize Agent replay buffer for current experiment with experience from previous experiment
        replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim,
                                                 agent_replay_size)  # Agent Replay Buffer for current experiment
        agent_replay_file_path = prev_exp_dir + "/replay_buffer/"     # Agent replay buffer location from prev. experiment
        replay_buffer.store_saved_data_into_replay(agent_replay_file_path)  # Fill experience from prev. stage into replay buffer

        # Initialize expert replay buffer, then generate expert pid data to fill it
        expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
        for shapes_to_load in requested_shapes:
            for orientations_to_load in requested_orientation_list:
                shape_replay_file_path = expert_replay_file_path + "/" + shapes_to_load + "/" + str(orientations_to_load) + "/replay_buffer/"
                # Load expert data from saved expert pid controller replay buffer
                expert_replay_buffer.store_saved_data_into_replay(shape_replay_file_path)

        if expert_replay_buffer.size == 0 or replay_buffer.size == 0:
            print("No experience in replay buffer! Quitting...")
            quit()

        # Save directory info for info file
        all_saving_dirs = setup_directories(env, exp_dir, expert_replay_file_path,
                                            agent_replay_file_path,
                                            prev_exp_dir, create_dirs=False)

        # Run experiment
        rl_experiment(policy, exp_num, exp_name, prev_exp_dir, requested_shapes, requested_orientation_list, all_saving_dirs, state_idx_arr=state_idx_arr)
    else:
        print("Invalid mode input")

    total_time_text = "\nTOTAL time: " + total_time.stop()
    print(total_time_text)
