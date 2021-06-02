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
from expert_data import GenerateExpertPID_JointVel, GenerateTestPID_JointVel, check_grasp
from timer import Timer
from pathlib import Path
import pathlib
import copy # For copying over coordinates
import glob # Used for getting saved policy filename
from expert_data import ExpertPIDController, get_action # Expert controller, used within evaluation

# Import plotting code from other directory
plot_path = os.getcwd() + "/plotting_code"
sys.path.insert(1, plot_path)
from heatmap_plot import generate_heatmaps, create_heatmaps, overlap_images
from boxplot_plot import generate_reward_boxplots
from heatmap_coords import add_heatmap_coords, filter_heatmap_coords, sort_coords_by_region
from evaluation_plots import reward_plot, generate_heatmaps_by_orientation_frame
from replay_stats_plot import get_selected_episode_metrics, actual_values_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')


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
    #         timestep_count = 0
    #         while not done:
    #             timestep_count += 1
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
    #         print("Eval timestep count: ",timestep_count)
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

def evaluate_coords_by_region(policy, all_hand_object_coords, variation_type, variation_heatmap_path, controller_type="policy"):
    """ Evaluate the policy within certain regions within the graspable area within the hand. Regions within
    the hand are determined by the Local coordinate frame. Plot and render a sample of success/failed coordinates.
    Policy: Policy to evaluate
    all_hand_object_coords: Coordinate dictionary with their representation within the local and global frame along with the
    hand orientation and evaluation results (reward value, success)
    variation_type: Input data type (Baseline, Baseline_HOV, etc.)
    variation_heatmap_path: Directory where the plots will be saved (Within the variation input directory)
    seed: seed value to conduct evaluation grasp trials with
    """
    frame = "local"
    requested_orientation = variation_type["requested_orientation"]
    if requested_orientation == "random":
        orientations_list = ["normal", "rotated", "top"]
    else:
        orientations_list = ["normal"]

    for orientation in orientations_list:
        heatmap_orient_dir = variation_heatmap_path + orientation + "/"
        hand_object_coords_dicts = [d for d in all_hand_object_coords if d["orientation"] == orientation]
        grasping_regions = ["extreme_left", "mid_left", "center", "mid_right", "extreme_right"]

        for region_name in grasping_regions:
            region_dicts = [d for d in hand_object_coords_dicts if d["local_coord_region"] == region_name]
            if len(region_dicts) > 0:
                region_dir = heatmap_orient_dir + "regions/" + region_name + "/"
                create_paths([region_dir])

                # Sample successful and failed points from evaluation
                success_dicts = [d for d in region_dicts if d["success"] is True]
                fail_dicts = [d for d in region_dicts if d["success"] is False]

                # Determine the point within the region with the most extreme value
                extreme_region_dict = copy.deepcopy(region_dicts[0])
                max_x_value = abs(extreme_region_dict["global_obj_coords"][0])
                
                for temp_dict in region_dicts:
                    if region_name == "center":
                        if abs(temp_dict["global_obj_coords"][0]) <= max_x_value:
                            extreme_region_dict = copy.deepcopy(temp_dict)
                    else:
                        if abs(temp_dict["global_obj_coords"][0]) >= max_x_value:
                            extreme_region_dict = copy.deepcopy(temp_dict)
                
                # Determine the hand position based on the most extreme x_value within the region
                wrist_coords = extreme_region_dict[frame + "_wrist_coords"]
                finger_coords = extreme_region_dict[frame + "_finger_coords"]

                # Sample 5 points from each region of success/failures to render
                for dict_list in [success_dicts, fail_dicts]:
                    for d_idx in range(min(1,len(dict_list))):
                        curr_dict = dict_list[d_idx]

                        # Evaluate policy with the most extreme coordinate point in the region and create a video rendering
                        region_eval_ret = eval_policy(policy, args.env_name, args.seed,
                                                      requested_shapes=variation_type["requested_shapes"],
                                                      requested_orientation=curr_dict["orientation"],
                                                      mode="eval", eval_episodes=1,
                                                      render_imgs=True,
                                                      start_pos=curr_dict["global_obj_coords"],
                                                      hand_rotation=curr_dict["hand_orient_variation"],
                                                      output_dir=region_dir,
                                                      with_noise=variation_type["with_orientation_noise"],
                                                      controller_type=controller_type)

                        all_action_values = region_eval_ret["all_action_values"]
                        policy_actions = all_action_values["policy_actions"]
                        render_file_dir = region_eval_ret["render_file_dir"]
                        range_of_episodes = np.arange(len(policy_actions))
                        for metric_idx in range(0, 3):
                            selected_ep_actions = get_selected_episode_metrics(range_of_episodes, policy_actions, metric_idx)
                            actual_values_plot(selected_ep_actions, 0, "Finger " + str(metric_idx+1) + " Velocity",
                                               "Policy Action Output: Finger " + str(metric_idx+1) + " Velocity",
                                               saving_dir=render_file_dir)

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

                create_heatmaps(success_x, success_y, fail_x, fail_y, total_x, total_y, "", orientation,
                                wrist_coords=wrist_coords, finger_coords=finger_coords, state_rep=frame,
                                saving_dir=region_dir, title_str="Input Variation: " + variation_type["variation_name"] + ", " + frame.capitalize() + " coord. frame, Region: " + region_name.capitalize())

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


# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, requested_shapes, requested_orientation, mode, with_noise, start_pos=None,hand_rotation=None,output_dir=None,eval_episodes=100, compare=False, render_imgs=False, controller_type="policy"):
    """ Evaluate policy in its given state over eval_episodes amount of grasp trials """
    num_success=0
    # Local representation of the object coordinates
    success_coords = {"x": [], "y": [], "orientation": []}
    fail_coords = {"x": [], "y": [], "orientation": []}
    # hand orientation types: NORMAL, Rotated (45 deg), Top (90 deg)

    # Initial (timestep = 0) transformation matrices (from Global to Local) for each episode
    all_hand_object_coords = []
    render_file_dir = output_dir

    # match timesteps to expert and pre-training
    max_num_timesteps = 30

    # Compare policy performance
    if compare:
        compare_test()

    # Make new environment for evaluation
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    # Generate randomized list of objects to select from
    eval_env.Generate_Latin_Square(eval_episodes,"eval_objects.csv",shape_keys=requested_shapes)

    avg_reward = 0.0
    # Reward data over each evaluation episode for boxplot
    all_ep_reward_values = {"total_reward": [], "finger_reward": [], "grasp_reward": [], "lift_reward": []}
    all_action_values = {"policy_actions":[],"actions_with_noise":[]}

    for i in range(eval_episodes):
        print("***Eval episode: ", i)
        success = 0
        state, done = eval_env.reset(shape_keys=requested_shapes, with_grasp=args.with_grasp_reward,hand_orientation=requested_orientation,mode="eval",env_name="eval_env",orient_idx=i,with_noise=with_noise,start_pos=start_pos,hand_rotation=hand_rotation), False

        # Initialize the controller for evaluation purposes
        if controller_type != "policy":
            controller = ExpertPIDController(state)

        # Record initial coordinate file path once shapes are generated
        all_saving_dirs["eval_init_coord_file_path"] = eval_env.get_coords_filename()
        cumulative_reward = 0
        # Sets number of timesteps per episode (counted from each step() call)
        eval_env._max_episode_steps = max_num_timesteps

        # Global object and hand coordinates
        hand_object_coords = {"local_obj_coords": [], "global_obj_coords": [], "local_to_global_obj_coords": [], "local_wrist_coords": [],
                              "global_wrist_coords": [], "local_to_global_wrist_coords": [], "global_to_local_transf": [], "orientation": [],
                              "hand_orient_variation": [], "local_finger_coords": [], "global_finger_coords": [], "local_to_global_finger_coords": [], "local_coord_region": "None", "total_episode_reward": 0, "success": 0}
        global_to_local_transf = eval_env.Tfw
        hand_orient_variation = eval_env.hand_orient_variation

        # Local finger positions (within the state)
        local_state = eval_env.get_obs_from_coord_frame(coord_frame="local")
        global_to_local_transf = eval_env.Tfw
        local_finger_coords = local_state[0:18] # Finger 1, 2, 3 proximal (x,y,z) coords, followed by distal coords
        local_wrist_coords = local_state[18:21]
        local_obj_coords = local_state[21:24]

        # Global hand and object positions
        global_state = eval_env.get_obs_from_coord_frame(coord_frame="global")
        global_to_local_transf = eval_env.Tfw
        global_finger_coords = global_state[0:18]
        global_wrist_coords = global_state[18:21]
        global_obj_coords = global_state[21:24]

        # Get the environment's current object coordinates
        env_global_obj_coords = np.array(eval_env.get_obj_coords())
        env_global_wrist_coords = eval_env.wrist_pose

        #print("Testing the Finger1 proximal coordinates (Global-->Local transformation)")
        #test_global_to_local_transformation(local_finger_coords[0:3], global_finger_coords[0:3], None, global_to_local_transf)
        #print("Testing the Wrist coordinates (Global-->Local transformation)")
        #test_global_to_local_transformation(local_wrist_coords, global_wrist_coords, env_global_wrist_coords, global_to_local_transf)
        #print("Testing the Object coordinates (Global-->Local transformation)")
        #test_global_to_local_transformation(local_obj_coords, global_obj_coords, env_global_obj_coords, global_to_local_transf)

        timestep_count = 0
        prev_state_lift_check = None
        curr_state_lift_check = state
        check_for_lift = True
        ready_for_lift = False
        skip_num_ts = 6
        curr_reward = 0

        # Cumulative reward over single episode
        ep_total_reward = 0
        ep_finger_reward = 0
        ep_grasp_reward = 0
        ep_lift_reward = 0

        # Actions taken by the policy over an episode
        episode_actions = []
        episode_actions_with_noise = []

        # Beginning of episode time steps, done is max timesteps or lift reward achieved
        while not done:
            #eval_env.render()
            #print("rendering...")

            timestep_count += 1
            if timestep_count < skip_num_ts:
                wait_for_check = False
            else:
                wait_for_check = True
            ##make it modular
            ## If None, skip
            if prev_state_lift_check is None:
                f_dist_old = None
            else:
                f_dist_old = prev_state_lift_check[9:17]
            f_dist_new = curr_state_lift_check[9:17]

            if check_for_lift and wait_for_check:
                [ready_for_lift, _] = check_grasp(f_dist_old, f_dist_new)

            #####
            # Not ready for lift, continue agent grasping following the policy
            if not ready_for_lift:
                if controller_type == "policy":
                    # Finger velocities
                    policy_action = policy.select_action(np.array(state[0:82])) # Due to the sigmoid should be between (0,max_action)
                    wrist_action = np.array([0])
                    action = np.concatenate((wrist_action, policy_action)) # If not ready for lift, wrist should always be 0
                else:
                    #print("Getting the action from the "+controller_type+" controller!!")
                    # Get the action from the controller (controller_type: naive, position-dependent)
                    # Finger velocities
                    controller_action = get_action(obs=np.array(state[0:82]), lift_check=ready_for_lift, controller=controller, env=eval_env, pid_mode=controller_type)
                    wrist_action = np.array([0])
                    action = np.concatenate((wrist_action, controller_action))  # The wrist velocity gets set within get_action

                eval_env.set_with_grasp_reward(args.with_grasp_reward)
                next_state, reward, done, info = eval_env.step(action) # Step action takes in the wrist velocity plus the finger velocities
                cumulative_reward += reward
                avg_reward += reward
                curr_reward = reward

                # Cumulative reward
                ep_total_reward += reward
                ep_finger_reward += info["finger_reward"]
                ep_grasp_reward += info["grasp_reward"]
                ep_lift_reward += info["lift_reward"]

                # Tracking actions with and without noise
                episode_actions.append(policy_action)
                action_with_noise = (policy_action + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(0, max_action)
                episode_actions_with_noise.append(action_with_noise)

            else:  # Make it happen in one time step
                next_state, reward, done, info,  cumulative_reward = eval_lift_hand(eval_env, cumulative_reward,
                                                                                 curr_reward)
                #print("Ready for lift < 0.035, np.max(np.array(obs[41:46])): ", np.max(np.array(state[41:46])))
                #print("Ready for lift < 0.015, np.max(np.array(obs[35:40])): ", np.max(np.array(state[35:40])))
                if done:
                    avg_reward += reward

                    # Cumulative reward
                    ep_total_reward += reward
                    ep_finger_reward += info["finger_reward"]
                    ep_grasp_reward += info["grasp_reward"]
                    ep_lift_reward += info["lift_reward"]
                check_for_lift = False

            if render_imgs is True:
                # Record an episode every 10 episodes
                if ready_for_lift:
                    action_str = "Constant lift action by controller"
                else:
                    action_str = "Policy Action Output (rad/sec):\nWrist Velocity: {}\nFinger 1 Velocity: {:.3f}\nFinger 2 Velocity: {:.3f}\nFinger 3 Velocity: {:.3f}".format(action[0], action[1],action[2],action[3])
                if done:
                    final_success = reward
                else:
                    final_success = None
                if i % 10 == 0:
                    # Set the info to be displayed in episode rendering based on current hand/object status
                    action_str = action_str + "\nObject Position (local x,y,z): {:.3f}, {:.3f}, {:.3f}\nReward: {}".format(local_obj_coords[0],local_obj_coords[1],local_obj_coords[2],reward)
                    render_file_dir = eval_env.render_img(text_overlay=action_str, episode_num=i,
                                               timestep_num=timestep_count,
                                               obj_coords=local_obj_coords, saving_dir=output_dir, final_episode_type=final_success)

            #####
            if reward >= 50:
                success = 1

            state = next_state
            prev_state_lift_check = curr_state_lift_check
            curr_state_lift_check = state

        # End of episode, record findings
        all_ep_reward_values["total_reward"].append(ep_total_reward)
        all_ep_reward_values["finger_reward"].append(ep_finger_reward)
        all_ep_reward_values["grasp_reward"].append(ep_grasp_reward)
        all_ep_reward_values["lift_reward"].append(ep_lift_reward)

        # Record all actions taken by the policy (with and without noise)
        episode_actions = np.stack(episode_actions, axis=0)
        episode_actions_with_noise = np.stack(episode_actions_with_noise, axis=0)

        all_action_values["policy_actions"].append(episode_actions)
        all_action_values["actions_with_noise"].append(episode_actions_with_noise)

        num_success += success

        # Add heatmap coordinates
        orientation = eval_env.get_orientation()
        ret = add_heatmap_coords(success_coords, fail_coords, orientation, local_obj_coords, success)
        success_coords = copy.deepcopy(ret["success_coords"])
        fail_coords = copy.deepcopy(ret["fail_coords"])

        # Save the hand-object coordinates to track the transformations
        hand_object_coords["local_obj_coords"] = local_obj_coords
        hand_object_coords["global_obj_coords"] = global_obj_coords
        hand_object_coords["local_wrist_coords"] = local_wrist_coords
        hand_object_coords["local_finger_coords"] = local_finger_coords
        hand_object_coords["global_finger_coords"] = global_finger_coords
        hand_object_coords["global_wrist_coords"] = global_wrist_coords
        hand_object_coords["global_to_local_transf"] = global_to_local_transf
        hand_object_coords["orientation"] = orientation
        hand_object_coords["hand_orient_variation"] = hand_orient_variation
        hand_object_coords["total_episode_reward"] = ep_total_reward
        hand_object_coords["success"] = bool(success)

        # Local to Global coordinate conversion (from our observation)
        local_to_global_temp = np.append(local_obj_coords, 1)
        local_to_global_temp = np.matmul(np.linalg.inv(global_to_local_transf), local_to_global_temp)
        local_to_global_obj_coords = local_to_global_temp[0:3].tolist()

        hand_object_coords["local_to_global_wrist_coords"] = local_to_global_transf(local_wrist_coords, global_to_local_transf)
        hand_object_coords["local_to_global_obj_coords"] = local_to_global_transf(local_obj_coords, global_to_local_transf)
        for coord_idx in range(0,len(local_finger_coords),3):
            transf_coords = local_to_global_transf(local_finger_coords[coord_idx:coord_idx+3], global_to_local_transf)
            hand_object_coords["local_to_global_finger_coords"].extend(transf_coords)

        all_hand_object_coords.append(hand_object_coords)

    avg_reward /= eval_episodes

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

    ret = {"avg_reward": avg_reward, "avg_rewards": avg_rewards, "all_ep_reward_values": all_ep_reward_values, "all_action_values" : all_action_values, "num_success": num_success, "success_coords": success_coords, "fail_coords": fail_coords, "all_hand_object_coords": all_hand_object_coords, "render_file_dir":render_file_dir}
    return ret


def lift_hand(env_lift, tot_reward):
    """ Lift hand with set action velocities
    env_lift: Mujoco environment
    tot_reward: Total cumulative reward
    """
    # action only used to move hand, not recorded in replay buffer and is NOT used to update policy
    action = np.array([wrist_lift_velocity, finger_lift_velocity, finger_lift_velocity,
                       finger_lift_velocity])
    env_lift.with_grasp_reward = args.with_grasp_reward
    next_state, reward, done, info = env_lift.step(action)
    if done:
        # accumulate or replace?
        old_reward = replay_buffer.replace(reward, done)
        tot_reward = tot_reward - old_reward + reward

    return next_state, reward, done, info, tot_reward


def eval_lift_hand(env_lift, tot_reward, curr_reward):
    """ Lift hand with set action velocities within evaluation environment
    env_lift: Mujoco environment
    tot_reward: Total cumulative reward
    curr_reward: Current time step reward
    """
    # action only used to move hand, not recorded in replay buffer and is NOT used to update policy
    action = np.array([wrist_lift_velocity, finger_lift_velocity, finger_lift_velocity,
                       finger_lift_velocity])
    env_lift.with_grasp_reward = args.with_grasp_reward
    next_state, reward, done, info = env_lift.step(action)
    if done:
        tot_reward = tot_reward - curr_reward + reward

    return next_state, reward, done, info, tot_reward


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


def update_policy(policy, evaluations, episode_num, num_episodes, prob,
                  type_of_training, all_saving_dirs, max_num_timesteps=30):
    """ Update policy network based on expert or agent step, evaluate every eval_freq episodes
    policy: policy to be updated
    evaluations: Output average reward list for plotting
    episode_num: Current episode
    num_episodes: Max number of episodes to update over
    writer: Tensorboard writer (avg. reward, loss, etc.)
    prob: Probability (proportion) of sampling from expert replay buffer
    type_of_training: Based on training mode ("pre-train", "eval", "test", etc.)
    max_num_timesteps: Maximum number of time steps within a RL episode
    """

    # Initialize and copy current policy to be the best policy
    best_policy = DDPGfD.DDPGfD()
    best_policy.copy(policy)

    # Get saving file paths from dictionary
    saving_dir = all_saving_dirs["saving_dir"]
    replay_dir = all_saving_dirs["replay_buffer"]

    # Initialize OU noise, added to action output from policy
    noise = OUNoise(4)
    noise.reset()
    expl_noise = OUNoise(4, sigma=0.001)
    expl_noise.reset()

    # Heatmap initial object coordinates for evaluation plots
    eval_success_coords = {"x": [], "y": [], "orientation": []}
    eval_fail_coords = {"x": [], "y": [], "orientation": []}
    # hand orientation types: NORMAL, Rotated (45 deg), Top (90 deg)
    
    # Number of successful/failed initial object coordinates from evaluation over the total # of grasp trials
    eval_num_success = 0
    eval_num_fail = 0
    eval_num_total = eval_num_success + eval_num_fail

    # Stores reward boxplot data, Average reward per evaluation episodes
    finger_reward = [[]]
    grasp_reward = [[]]
    lift_reward = [[]]
    total_reward = [[]]
    eval_action = [[]]
    eval_action_with_noise = [[]]

    # At episode 0, the target networks == current netoworks, so the network loss is 0
    actor_loss = 0
    critic_loss = 0
    critic_L1loss = 0
    critic_LNloss = 0

    # Tensorboard writer
    writer = SummaryWriter(logdir=all_saving_dirs["tensorboard_dir"])

    for _ in range(num_episodes):
        env = gym.make(args.env_name)

        # Max number of time steps to match the expert replay grasp trials
        env._max_episode_steps = max_num_timesteps

        # Fill training object list using latin square
        if env.check_obj_file_empty("objects.csv"):
            env.Generate_Latin_Square(args.max_episode, "objects.csv", shape_keys=requested_shapes)
        state, done = env.reset(shape_keys=requested_shapes, with_grasp=args.with_grasp_reward,env_name="env", hand_orientation=requested_orientation,
                                mode=args.mode, with_noise=with_orientation_noise), False

        # Set whether or not to use grasp reward
        env.set_with_grasp_reward(args.with_grasp_reward)
        # Record initial coordinate file path once shapes are generated
        all_saving_dirs["train_init_coord_file_path"] = env.get_coords_filename()

        prev_state_lift_check = None
        curr_state_lift_check = state

        noise.reset()
        expl_noise.reset()
        episode_reward = 0
        obj_coords = env.get_obj_coords()
        # Local coordinate conversion
        obj_local = np.append(obj_coords,1)
        obj_local = np.matmul(env.Tfw,obj_local)
        local_obj_coords = obj_local[0:3]

        replay_buffer.add_episode(1)
        # Add orientation noise to be recorded by replay buffer
        orientation_idx = env.get_orientation_idx()
        replay_buffer.add_orientation_idx_to_replay(orientation_idx)

        # Determine the expert replay buffer to be used based on the selected shape
        current_object = env.get_random_shape()
        expert_replay_buffer = expert_buffers[current_object]

        timestep = 0
        replay_buffer_recorded_ts = 0
        check_for_lift = True
        ready_for_lift = False
        skip_num_ts = 6
        lift_success = False # Set based on lift reward output at end of episode

        print(type_of_training, episode_num)

        # Beginning of time steps within episode
        while not done:
            timestep = timestep + 1
            if timestep < skip_num_ts:
                wait_for_check = False
            else:
                wait_for_check = True
            ##make it modular
            ## If None, skip
            if prev_state_lift_check is None:
                f_dist_old = None
            else:
                f_dist_old = prev_state_lift_check[9:17]
            f_dist_new = curr_state_lift_check[9:17]

            if check_for_lift and wait_for_check:
                [ready_for_lift, _] = check_grasp(f_dist_old, f_dist_new)

            # Follow policy until ready for lifting, then switch to set controller
            if not ready_for_lift:
                policy_action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(0, max_action)
                wrist_action = np.array([0])
                action = np.concatenate((wrist_action, policy_action)) # Policy outputs the finger velocities - the wrist velocity will be 0 until ready for lift

                # Perform action obs, total_reward, done, info
                env.set_with_grasp_reward(args.with_grasp_reward)
                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state[0:82], policy_action, next_state[0:82], reward, float(done))
                replay_buffer_recorded_ts += 1
                episode_reward += reward

            else:  # Make it happen in one time step
                next_state, reward, done, info, episode_reward = lift_hand(env, episode_reward)
                check_for_lift = False
                # Determine successful grasp based on lift reward
                if info["lift_reward"] > 0:
                    lift_success = True
            state = next_state

            prev_state_lift_check = curr_state_lift_check
            curr_state_lift_check = state

        replay_buffer.add_episode(0)  # Add entry for new episode

        # Remove any invalid episodes (episodes shorter than n-step length for policy training)
        episode_len = replay_buffer_recorded_ts # Number of timesteps within the episode recorded by replay buffer
        if episode_len - replay_buffer.n_steps <= 1:
            replay_buffer.remove_episode(-1)  # If episode is invalid length (less that n-steps), remove it

        # Train agent after collecting sufficient data:
        if episode_num >= args.update_after: # Update policy after 100 episodes (have enough experience in agent replay buffer)
            #if episode_num % args.update_freq: # Update every 4 episodes
            for update_count in range(args.update_num): # Number of times to update the policy
                if args.batch_size == 0:
                    # Single episode training using full trajectory
                    actor_loss, critic_loss, critic_L1loss, critic_LNloss = policy.train(env._max_episode_steps,
                                                                                     expert_replay_buffer,
                                                                                     replay_buffer, prob)
                else:
                    # Batch training using n-steps
                    actor_loss, critic_loss, critic_L1loss, critic_LNloss = policy.train_batch(env._max_episode_steps, episode_num, update_count,
                                                                                         expert_replay_buffer,
                                                                                         replay_buffer)

        # Evaluation and recording data for tensorboard
        if episode_num+1 == num_episodes or (episode_num) % args.eval_freq == 0:
            print("EVALUATING EPISODE AT: ",episode_num)
            print("Evaluating with "+str(args.eval_num)+" grasping trials")
            eval_ret = eval_policy(policy, args.env_name, args.seed, requested_shapes, requested_orientation,
                                   mode=args.mode, eval_episodes=args.eval_num, with_noise=with_orientation_noise, render_imgs=args.render_imgs)  # , compare=True)
            # Heatmap data - object starting coordinates for evaluation
            eval_success_coords = copy.deepcopy(eval_ret["success_coords"])
            eval_fail_coords = copy.deepcopy(eval_ret["fail_coords"])

            # Records the number of successful and failed coordinates from evaluation
            eval_num_success = len(eval_success_coords["x"])
            eval_num_fail = len(eval_success_coords["x"])
            eval_num_total = eval_num_success + eval_num_fail

            # Cumulative (over timesteps) reward data from each evaluation episode for boxplot
            all_ep_reward_values = eval_ret["all_ep_reward_values"]
            all_action_values = eval_ret["all_action_values"]

            # Plot tensorboard metrics for learning analysis (average reward, loss, etc.)
            writer = write_tensor_plot(writer,episode_num,eval_ret["avg_reward"],eval_ret["avg_rewards"],actor_loss,critic_loss,critic_L1loss,critic_LNloss,policy.current_expert_proportion)

            # Insert boxplot code reference
            finger_reward[-1].append(all_ep_reward_values["finger_reward"])
            grasp_reward[-1].append(all_ep_reward_values["grasp_reward"])
            lift_reward[-1].append(all_ep_reward_values["lift_reward"])
            total_reward[-1].append(all_ep_reward_values["total_reward"])
            eval_action[-1].append(all_action_values["policy_actions"])
            eval_action_with_noise[-1].append(all_action_values["actions_with_noise"])

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
            else:
                policy.copy(best_policy)
                print("Evaluation episode: " + str(episode_num) + " COPYING Best Policy to be the CURRENT policy")

        # Save coordinates every 200 episodes
        if episode_num+1 == num_episodes or (episode_num) % args.save_freq == 0:
            print("Saving heatmap data at: ", all_saving_dirs["heatmap_dir"])
            # Filter heatmap coords by success/fail, orientation type, and save to appropriate place
            filter_heatmap_coords(eval_success_coords, eval_fail_coords, episode_num, all_saving_dirs["heatmap_dir"])
            # Reset eval coords for next batch
            eval_success_coords = {"x": [], "y": [], "orientation": []}
            eval_fail_coords = {"x": [], "y": [], "orientation": []}

            print("Saving boxplot data at: ", all_saving_dirs["boxplot_dir"])
            np.save(all_saving_dirs["boxplot_dir"] + "/finger_reward_" + str(episode_num),finger_reward)
            np.save(all_saving_dirs["boxplot_dir"] + "/grasp_reward_" + str(episode_num),grasp_reward)
            np.save(all_saving_dirs["boxplot_dir"] + "/lift_reward_" + str(episode_num),lift_reward)
            np.save(all_saving_dirs["boxplot_dir"] + "/total_reward_" + str(episode_num),total_reward)

            # Save the finger velocities for plotting and evaluation
            actions_path = all_saving_dirs["results_saving_dir"] + "/eval_actions/policy_actions_" + str(episode_num)
            actions_with_noise_path = all_saving_dirs["results_saving_dir"] + "/eval_actions/actions_with_noise_" + str(episode_num)
            create_paths([all_saving_dirs["results_saving_dir"] + "/eval_actions/"])
            np.save(actions_path, all_action_values["policy_actions"])
            np.save(actions_with_noise_path, all_action_values["actions_with_noise"])

            finger_reward = [[]]
            grasp_reward = [[]]
            lift_reward = [[]]
            total_reward = [[]]
            eval_action = [[]]
            eval_action_with_noise = [[]]

        episode_num += 1

    # Training is complete, now save policy and replay buffer
    # Save policy
    print("Saving the BEST policy...")
    best_policy.save(all_saving_dirs["model_save_path"])
    print("Saved policy at: ", all_saving_dirs["model_save_path"])

    print("Saving Agent replay buffer experience...")
    agent_replay_save_path = replay_buffer.save_replay_buffer(replay_dir)
    print("Saved agent replay buffer at: ",agent_replay_save_path)

    return evaluations, episode_num, eval_num_success, eval_num_fail


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


def train_policy(policy, expert_buffers, replay_buffer, tot_episodes, tr_prob, all_saving_dirs):
    """ Train the policy over a number of episodes, sampling from experience
    tot_episodes: Total number of episodes to update policy over
    tr_prob: Probability of sampling from expert replay buffer within training
    """
    evals = []
    curr_episode = 0         # Counts number of episodes done within training

    # Begin training updates
    evals, curr_episode, eval_num_success, eval_num_fail = \
        update_policy(policy, evals, curr_episode, tot_episodes, tr_prob,"TRAIN", all_saving_dirs)

    eval_num_total = eval_num_success + eval_num_fail

    return all_saving_dirs["model_save_path"], eval_num_success, eval_num_total


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


def generate_output(text, orientations_list, num_success, num_total, all_saving_dirs, plot_type="eval"):
    """ Generate heatmaps, boxplots, and output info file """
    # Produce plots

    # Train Heatmap
    if os.path.isdir(all_saving_dirs["heatmap_dir"]) is True:
        print("Generating heatmaps...")
        for orientation in orientations_list:
            plot_output_dir = all_saving_dirs["heatmap_dir"] + orientation + "/"
            generate_heatmaps(plot_type=plot_type, orientation=str(orientation), data_dir=plot_output_dir,
                              saving_dir=plot_output_dir,max_episodes=args.max_episode, saving_freq=args.save_freq)
    else:
        print("Heatmap dir NOT found: ", all_saving_dirs["heatmap_dir"])

    if os.path.isdir(all_saving_dirs["boxplot_dir"]) is True:
        print("Generating boxplots...")
        # Boxplot evaluation reward
        for orientation in orientations_list:
            plot_output_dir = all_saving_dirs["boxplot_dir"] + orientation + "/"
            generate_reward_boxplots(plot_type=plot_type,orientation=str(orientation), data_dir=plot_output_dir,
                                     saving_dir=plot_output_dir, tot_episodes=args.max_episode, saving_freq=args.save_freq, eval_freq=args.eval_freq)
    else:
        print("Boxplot dir NOT found: ", all_saving_dirs["boxplot_dir"])

    print("Writing to experiment info file...")
    if all_saving_dirs is not None:
        create_info_file(num_success, num_total, all_saving_dirs, text)


def rl_experiment(policy, exp_num, exp_name, prev_exp_dir, requested_shapes, requested_orientation_list, all_saving_dirs):
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
    policy, train_model_save_path, eval_num_success, eval_num_total = train_policy(policy, expert_replay_buffer, replay_buffer,  args.max_episode, args.expert_prob, all_saving_dirs)
    print("Experiment ", exp_num, ", ", exp_name, " policy saved at: ", train_model_save_path)

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
    generate_output(text=text, orientations_list=requested_orientation_list, num_success=eval_num_success, num_total=eval_num_total, all_saving_dirs=all_saving_dirs)

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
    parser.add_argument("--max_episode", default=20000, type=int)       # Max time steps to run environment for
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
    parser.add_argument("--pretrain_policy_path", type=str, action='store', default="./experiments/pre-train/Critic_sigmoid_InOrder/policy/pre-train_DDPGfD_kinovaGrip_04_29_21_1223/") # Path to the pre-trained policy to be loaded
    parser.add_argument("--test_policy_path", type=str, action='store', default="./experiments/pre-train/Critic_sigmoid_InOrder/policy/pre-train_DDPGfD_kinovaGrip_04_29_21_1223/") # Path of the policy to be tested
    parser.add_argument("--test_policy_name", type=str, action='store', default="") # Test policy name for clear plotting
    parser.add_argument("--with_orientation_noise", type=str, action='store', default="False") # Set to true to sample initial hand-object coordinates from with_noise/ dataset
    parser.add_argument("--controller_type", type=str, action='store', default="policy") # Determine the type of controller to use for evaluation (policy OR naive, expert, position-dependent)

    args = parser.parse_args()
    return args


def test_policy_models_match(current_model, compare_model=None, compare_model_filepath=None):
    """ Check that the input models match. This can check that we have loaded in the trained model correctly. """

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

    # Set dimensions for state and action spaces - policy initialization
    state_dim = 82  # State dimension dependent on the length of the state space
    action_dim = 3 #env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_action_trained = env.action_space.high  # a vector of max actions
    n = 5   # n step look ahead for the policy
    max_q_value = 50  # Should match the maximum reward value
    wrist_lift_velocity = 0.6
    min_velocity = 0.5  # Minimum velocity value for fingers or wrist
    finger_lift_velocity = min_velocity

    ''' Set values from command line arguments '''
    requested_shapes = args.shapes                   # Sets list of desired objects for experiment
    requested_shapes = requested_shapes.split(',')
    requested_orientation = args.hand_orientation   # Set the desired hand orientation (normal or random)
    expert_replay_size = args.expert_replay_size    # Number of expert episodes for expert the replay buffer
    agent_replay_size = args.agent_replay_size      # Maximum number of episodes to be stored in agent replay buffer
    max_num_timesteps = 30     # Maximum number of time steps within an episode

    # If experiment number is selected, set mode to experiment (in case the mode has been set to train by default)
    if args.exp_num is not None:
        args.mode = "experiment"

    # Set requested_orientation_list for directory creation, plotting and reference
    if requested_orientation == "random":
        requested_orientation_list = ["normal", "rotated", "top"]
    else:
        requested_orientation_list = ["normal"]

    # Fill pre-training object list using latin square method
    env.Generate_Latin_Square(args.max_episode,"objects.csv", shape_keys=requested_shapes)

    kwargs = {
        "state_dim": state_dim,
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

    if args.tensorboardindex is None:
        args.tensorboardindex = "%s_%s" % (args.policy_name, args.mode)
    args.tensorboardindex = args.tensorboardindex[:30]  # Keep the tensorboard name at a max size of 30 characters

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
        param_text += "Expert Sampling Probability: "+ str(args.expert_prob) + "\n"
        param_text += "Grasp Reward: "+ str(args.with_grasp_reward) + "\n"
        param_text += "Save frequency: "+ str(args.save_freq) + "\n"
        param_text += "Policy update after: "+ str(args.update_after) + "\n"
        param_text += "Policy update frequency: "+ str(args.update_freq) + "\n"
        param_text += "Policy update Amount: "+ str(args.update_num) + "\n"
        if args.mode != "position-dependent" and args.mode != "naive" and args.mode != "combined":
            param_text += "Generating " + str(args.max_episode) + " episodes!"
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
    expert_replay_file_path = experiment_dir + "naive/" + noise_str + "/" + with_grasp_str + "/"
    print("** expert_replay_file_path: ", expert_replay_file_path)

    ## Agent Replay Buffer ##
    agent_replay_file_path = "./experiments/pre-train/Critic_sigmoid_InOrder/replay_buffer/replay_buffer_04_29_21_1223/" # FILL WITH AGENT REPLAY FROM PRETRAINING

    ## Pre-trained Policy ##
    # Default pre-trained policy file path
    pretrain_model_save_path = args.pretrain_policy_path
    test_policy_path = args.test_policy_path
    test_policy_name = args.test_policy_name

    # Initialize timer to analyze run times
    total_time = Timer()
    total_time.start()

    # Determine replay buffer/policy function calls based on mode (expert, pre-train, train, rand_train, test)
    # Generate expert data based on Expert nudge controller only
    if args.mode == "position-dependent":
        print("MODE: Expert ONLY")
        # Initialize expert replay buffer, then generate expert pid data to fill it
        expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
        expert_replay_buffer, expert_replay_file_path, expert_data_dir, info_file_text, num_success, num_total, coord_filepath = GenerateExpertPID_JointVel(expert_replay_size, requested_shapes, requested_orientation, saving_dir, replay_buffer=expert_replay_buffer, with_grasp=args.with_grasp_reward, with_noise=with_orientation_noise, save=True, render_imgs=args.render_imgs, pid_mode="position-dependent")
        print("Expert ONLY expert_data_dir: ", expert_data_dir)
        print("Expert ONLY expert_replay_file_path: ",expert_replay_file_path, "\n", expert_replay_buffer)

        # Create directories where information will be saved
        all_saving_dirs = setup_directories(env, expert_data_dir, "None", "None", "None", create_dirs=False)
        all_saving_dirs["controller_init_coord_file_path"] = coord_filepath

        generate_output(text="\nPARAMS: \n"+param_text+info_file_text, orientations_list=requested_orientation_list, num_success=num_success, num_total=num_total, all_saving_dirs=all_saving_dirs, plot_type=None)

    # Generate expert data based on Naive controller only
    elif args.mode == "naive":
        print("MODE: Constant-Speed")
        # Create directories where information will be saved
        all_saving_dirs = setup_directories(env, saving_dir, "None", "None", "None",create_dirs=True)

        # Initialize expert replay buffer, then generate expert pid data to fill it
        expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
        expert_replay_buffer, expert_replay_file_path, expert_data_dir, info_file_text, num_success, num_total, coord_filepath = GenerateExpertPID_JointVel(expert_replay_size, requested_shapes, requested_orientation, all_saving_dirs, replay_buffer=expert_replay_buffer, with_grasp=args.with_grasp_reward, with_noise=with_orientation_noise, save=True, render_imgs=args.render_imgs, pid_mode="naive")
        print("Constant-speed expert_data_dir: ", expert_data_dir)
        print("Constant-speed expert_replay_file_path: ",expert_replay_file_path, "\n", expert_replay_buffer)

        all_saving_dirs["controller_init_coord_file_path"] = coord_filepath

        generate_output(text="\nPARAMS: \n"+param_text+info_file_text, orientations_list=requested_orientation_list, num_success=num_success, num_total=num_total, all_saving_dirs=all_saving_dirs,plot_type=None)

    # Generate expert data based on interpolating naive and expert strategies
    elif args.mode == "combined":
        print("MODE: Expert (Interpolation)")
        # Initialize expert replay buffer, then generate expert pid data to fill it
        expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)

        expert_replay_buffer, expert_replay_file_path, expert_data_dir, info_file_text, num_success, num_total, coord_filepath = GenerateExpertPID_JointVel(expert_replay_size, requested_shapes, requested_orientation, saving_dir, replay_buffer=expert_replay_buffer, with_grasp=args.with_grasp_reward, with_noise=with_orientation_noise, save=True, render_imgs=args.render_imgs, pid_mode="combined")
        print("Expert (Interpolation) expert_data_dir: ", expert_data_dir)
        print("Expert (Interpolation) expert_replay_file_path: ",expert_replay_file_path, "\n", expert_replay_buffer)

        # Create directories where information will be saved
        all_saving_dirs = setup_directories(env, expert_data_dir, expert_replay_file_path,
                                            "None", "None",create_dirs=False)
        all_saving_dirs["controller_init_coord_file_path"] = coord_filepath

        # Generate plots and info file
        generate_output(text="\nPARAMS: \n"+param_text+info_file_text, orientations_list=requested_orientation_list, num_success=num_success, num_total=num_total, all_saving_dirs=all_saving_dirs, plot_type=None)

    # Pre-train policy using expert data, save pre-trained policy for use in training
    elif args.mode == "pre-train":
        print("MODE: Pre-train")
        print("Expert replay Buffer: ", expert_replay_file_path)
        agent_replay_file_path = None
        print("Agent replay Buffer: ", agent_replay_file_path)

        # Initialize Queue Replay Buffer: replay buffer manages its size like a queue, popping off the oldest episodes
        replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, agent_replay_size)

        # Initialize expert replay buffer, then generate expert pid data to fill it
        #expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)

        #print("Loading expert replay buffer: ", expert_replay_file_path)
        #replay_text = expert_replay_buffer.store_saved_data_into_replay(expert_replay_file_path)

        # Determine the expert replay buffer(s) to be used based on the requested shapes
        expert_buffers = {}
        for shape_to_load in requested_shapes:
            expert_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
            shape_replay_file_path = expert_replay_file_path + shape_to_load + "/" + str(requested_orientation) + "/replay_buffer/"
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
        pretrain_model_save_path, eval_num_success, eval_num_total = train_policy(policy, expert_buffers, replay_buffer, args.max_episode, args.expert_prob,all_saving_dirs)
        train_time_text = "\nTRAIN time: \n" + train_time.stop()
        print(train_time_text)
        print("\nTrain complete! Now saving...")
        agent_replay_file_path = all_saving_dirs["replay_buffer"] + "/"

        # Create plots and info file
        generate_output(text="\nPARAMS: \n"+param_text+train_time_text+"\n"+replay_text, orientations_list=requested_orientation_list, num_success=eval_num_success, num_total=eval_num_total, all_saving_dirs=all_saving_dirs)

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
            replay_buffer.store_saved_data_into_replay(agent_replay_file_path)

        # Determine the expert replay buffer(s) to be used based on the requested shapes
        expert_buffers = {}
        for shape_to_load in requested_shapes:
            expert_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
            shape_replay_file_path = expert_replay_file_path + shape_to_load + "/" + str(requested_orientation) + "/replay_buffer/"
            # Load expert data from saved expert pid controller replay buffer
            print("Loading expert replay buffer: ", shape_replay_file_path)
            replay_text = expert_buffer.store_saved_data_into_replay(shape_replay_file_path)
            expert_buffers[shape_to_load] = copy.deepcopy(expert_buffer)

        """
        # Test model saving is correct
        print("Checking LOADED pre-trained policy and SAVED training model match!!")
        match_check = test_policy_models_match(policy, compare_model_filepath=pretrain_model_save_path)
        if match_check is False:
            print("Policies do NOT match! Quitting!")
            quit()
        """

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
        policy.sampling_decay_rate = 0.2
        train_model_save_path, eval_num_success, eval_num_total = train_policy(policy, expert_buffers, replay_buffer,  args.max_episode, args.expert_prob,all_saving_dirs)
        train_time_text = "\nTRAIN time: \n" + train_time.stop()
        print(train_time_text)
        print("\nTrain complete!")

        # Create plots and info file
        generate_output(text="\nPARAMS: \n"+param_text+train_time_text, orientations_list=requested_orientation_list, num_success=eval_num_success, num_total=eval_num_total, all_saving_dirs=all_saving_dirs)

    # Train policy given randomly initialized policy
    elif args.mode == "rand_train":
        print("MODE: Train (Random init policy)")
        print("Expert replay Buffer: ", expert_replay_file_path)
        print("Agent replay Buffer: ", agent_replay_file_path)

        # Set to None to handle text info file output for expert/non-expert data generation
        info_file_text = "None"
        expert_output_data_dir = "None"
        num_success = "None"
        num_total = "None"

        # Initialize Queue Replay Buffer: replay buffer manages its size like a queue, popping off the oldest episodes
        replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, agent_replay_size)

        if expert_replay_size == 0:
            expert_replay_buffer = None
        else:
            # Initialize expert replay buffer, then generate expert pid data to fill it
            expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
            expert_replay_buffer, expert_replay_file_path, expert_output_data_dir, info_file_text, num_success, num_total, coord_filepath = GenerateExpertPID_JointVel(expert_replay_size, requested_shapes, requested_orientation, saving_dir, replay_buffer=expert_replay_buffer, with_grasp=args.with_grasp_reward, with_noise=with_orientation_noise, save=True, render_imgs=args.render_imgs, pid_mode="combined")

        # Model replay buffer saving file name
        replay_filename = saving_dir + "/replay_buffer" + datestr

        # Create directories where information will be saved
        all_saving_dirs = setup_directories(env, saving_dir, expert_replay_file_path, agent_replay_file_path, pretrain_model_save_path)

        # Train the policy and save it
        train_model_save_path, eval_num_success, eval_num_total = train_policy(policy, expert_replay_buffer, replay_buffer,  args.max_episode, args.expert_prob,all_saving_dirs)

        # Create plots and info file
        generate_output(text="\nPARAMS: \n"+param_text, orientations_list=requested_orientation_list, num_success=eval_num_success, num_total=eval_num_total, all_saving_dirs=all_saving_dirs)

    # Test policy over certain number of episodes -- In Progress
    elif args.mode == "eval":
        print("MODE: Evaluate")
        print("Policy: ", test_policy_path)
        print("Policy Name: ", test_policy_name)

        # Determine whether we're evaluating over the policies or comparing performance of a certain controller
        controller_type = str(args.controller_type)

        # Create directories where information will be saved
        all_saving_dirs = setup_directories(env, saving_dir, expert_replay_file_path, agent_replay_file_path, test_policy_path)

        # Evaluation variations
        Baseline = {"variation_name": "Baseline", "requested_shapes": ["CubeS"], "requested_orientation": "normal", "with_orientation_noise": False}
        Baseline_HOV = {"variation_name": "Baseline_HOV", "requested_shapes": ["CubeS"], "requested_orientation": "normal", "with_orientation_noise": True}
        Sizes_HOV = {"variation_name": "Sizes_HOV", "requested_shapes": ["CubeS","CubeM","CubeB"], "requested_orientation": "normal", "with_orientation_noise": True}
        Shapes_HOV = {"variation_name": "Shapes_HOV", "requested_shapes": ["CubeS", "CylinderS", "Vase2S"], "requested_orientation": "normal", "with_orientation_noise": True}

        variations = [Baseline, Baseline_HOV, Sizes_HOV, Shapes_HOV]

        max_episode = 4000
        eval_freq = 1000
        num_policies = int(max_episode / eval_freq) + 1

        if max_episode == 0:
            policy_eval_points = np.array([0])
        else:
            policy_eval_points = np.linspace(start=0, stop=max_episode, num=num_policies, dtype=int)
        # Currently this is for one policy, but it will be for each policy

        # All rewards (over each evaluation point) for each policy per variation type
        variation_rewards_per_policy = {}

        if controller_type == "policy":
            # Current policy we are evaluating over each evaluation point
            policies = ["Baseline", "Baseline_HOV", "Sizes_HOV", "Shapes_HOV"]
        else:
            policies = [controller_type]

        for policy_name in policies:
            current_policy_path = test_policy_path+"/"+policy_name+"/output/results/"

            # For each evaluation point, append the avg. reward from evaluating the policy
            # PER POLICY This will contain the current policy's list of vg. rewards over each evaluation point
            policy_rewards = {"Baseline": [], "Baseline_HOV": [], "Sizes_HOV": [],
                                 "Shapes_HOV": []}

            for idx in range(len(policy_eval_points)):

                if controller_type == "policy":
                    # Load policy from the evaluation point
                    ep_num = policy_eval_points[idx]

                    eval_point_policy_path = current_policy_path + "/policy_" + str(ep_num) + "/"
                    current_test_output_path = eval_point_policy_path + "output/"
                    create_paths([current_test_output_path])

                    print("Loading policy: ",eval_point_policy_path)
                    policy.load(eval_point_policy_path)  # Change to be complete, trained policy
                else:
                    eval_point_policy_path = current_policy_path + "/" + controller_type + "/"
                    current_test_output_path = eval_point_policy_path + "output/"
                    create_paths([current_test_output_path])
                    print("Using controller: ", controller_type)

                for variation_type in variations:
                    variation_name = variation_type["variation_name"]
                    variation_output_path = current_test_output_path + variation_type["variation_name"]
                    variation_heatmap_path = variation_output_path + "/heatmap/"
                    create_paths([variation_output_path, variation_heatmap_path])

                    print("Now evaluating: ", variation_type.items())
                    print("current_test_output_path: ",current_test_output_path)
                    print("variation_output_path: ", variation_output_path)

                    # Evaluate policy over certain number of episodes
                    eval_ret = eval_policy(policy, args.env_name, args.seed, requested_shapes=variation_type["requested_shapes"], requested_orientation=variation_type["requested_orientation"],
                                           mode=args.mode, eval_episodes=500, render_imgs=args.render_imgs, with_noise=variation_type["with_orientation_noise"])

                    # Append the average reward from evaluation to the specific variation type avg. reward list
                    policy_rewards[variation_name].append(eval_ret["avg_reward"])

                    # Heatmap data - object starting coordinates for evaluation
                    eval_success_coords = copy.deepcopy(eval_ret["success_coords"])
                    eval_fail_coords = copy.deepcopy(eval_ret["fail_coords"])

                    # Sorts coordinates by success/failure per hand orientation
                    filter_heatmap_coords(eval_success_coords, eval_fail_coords, "", variation_heatmap_path)

                    # All_hand_object_coords is a dictionary containing each hand and object coord. used within evaluation
                    all_hand_object_coords = eval_ret["all_hand_object_coords"]

                    # Save the hand and object coordinates -- within the current policy's variation folder (Ex: Policy_0/Baseline/)
                    dict_file = open(variation_output_path+"/all_hand_object_coords.csv", "w", newline='')
                    keys = all_hand_object_coords[0].keys()
                    dict_writer = csv.DictWriter(dict_file, keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(all_hand_object_coords)
                    dict_file.close()

                    # Generate heatmap plots based on each orientation type used in evaluation (and per coordinate frame - Local, Global, Local-->Global)
                    generate_heatmaps_by_orientation_frame(variation_type, all_hand_object_coords, variation_heatmap_path)

                    ## RENDER AND PLOT COORDINATES BY REGION
                    evaluate_coords_by_region(policy, all_hand_object_coords, variation_type, variation_heatmap_path, controller_type=controller_type)

            # AFTER we have gone through each evaluation point, record the current policy's rewards for each variation
            variation_rewards_per_policy[policy_name] = policy_rewards

        policy_colors = {"Baseline": "#808080", "Baseline_HOV": "black", "Sizes_HOV": "blue", "Shapes_HOV": "red"}
        # variation_input_policies format: For the current variation type, we get {policy_name: rewards}

        # Create a reward plot for each variation input type
        for variation_type in variations:
            variation_input_name = variation_type["variation_name"]
            variation_input_policies = {}

            # Get the reward data from each policy for the current variation_input_name
            for policy_name in policies:
                policy_rewards_dict = variation_rewards_per_policy[policy_name]
                # For the CURRENT VARIATION: {"Baseline": [reward, reward, ..]
                variation_input_policies[policy_name] = policy_rewards_dict[variation_input_name]

            # Save reward data to file
            dict_file = open(test_policy_path + "/" +str(variation_input_name) + "_policy_rewards.csv", "w", newline='')
            keys = variation_input_policies.keys()
            dict_writer = csv.DictWriter(dict_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows([variation_input_policies])
            dict_file.close()

            # Create a reward plot for variation_input_name
            reward_plot(policy_eval_points, variation_input_policies, variation_input_name, policy_colors, eval_freq=eval_freq, saving_dir=test_policy_path)


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
            shape_replay_file_path = expert_replay_file_path + "/" + shapes_to_load + "/" + str(requested_orientation) + "/replay_buffer/"
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
        rl_experiment(policy, exp_num, exp_name, prev_exp_dir, requested_shapes, requested_orientation_list, all_saving_dirs)
    else:
        print("Invalid mode input")

    total_time_text = "\nTOTAL time: " + total_time.stop()
    print(total_time_text)
