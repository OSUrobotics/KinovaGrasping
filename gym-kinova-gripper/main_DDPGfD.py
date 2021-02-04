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
from expert_data import store_saved_data_into_replay, GenerateExpertPID_JointVel, GenerateTestPID_JointVel, naive_check_grasp
from timer import Timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')


def save_coordinates(x,y,filename,episode_num):
    """ Save heatmap initial object position x,y coordinates
    x: initial object x-coordinate
    y: initial object y-coordinate
    filename: Location to save heatmap coordinates to
    """
    ep_str = ""
    if episode_num is not None:
        ep_str = "_"+str(episode_num)

    np.save(filename+"_x"+ep_str, x)
    np.save(filename+"_y"+ep_str, y)

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
    #         state, done = eval_env.reset(start_pos=[x,y],env_name="eval_env",shape_keys=requested_shapes,hand_orientation=requested_orientation,mode=mode), False
    #         success=0
    #         # Sets number of timesteps per episode (counted from each step() call)
    #         #eval_env._max_episode_steps = 200
    #         eval_env._max_episode_steps = max_num_timesteps
    #         while not done:
    #             action = GenerateTestPID_JointVel(np.array(state),eval_env)
    #             state, reward, done, _ = eval_env.step(action)
    #             avg_reward += reward
    #             cumulative_reward += reward
    #             if reward > 25:
    #                 success=1
    #         num_success[1]+=success
    #         print('PID net reward:',cumulative_reward)
    #         state, done = eval_env.reset(start_pos=[x,y],env_name="eval_env",shape_keys=requested_shapes,hand_orientation=requested_orientation,mode=mode), False
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

# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, requested_shapes, requested_orientation, mode, eval_episodes=100, compare=False):
    """ Evaluate policy in its given state over eval_episodes amount of grasp trials """
    num_success=0
    # Heatmap plot success/fail object coordinates
    seval_obj_posx = np.array([])
    feval_obj_posx = np.array([])
    seval_obj_posy = np.array([])
    feval_obj_posy = np.array([])
    total_evalx = np.array([])      # Total evaluation object coords
    total_evaly = np.array([])

    # match timesteps to expert and pre-training
    max_num_timesteps = 30

    # Folder to save heatmap coordinates
    evplot_saving_dir = "./eval_plots"
    if not os.path.isdir(evplot_saving_dir):
        os.mkdir(evplot_saving_dir)

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

    for i in range(eval_episodes):
        print("***Eval episode: ", i)
        success=0
        state, done = eval_env.reset(hand_orientation=requested_orientation,mode=args.mode,shape_keys=requested_shapes,env_name="eval_env"), False
        cumulative_reward = 0
        # Sets number of timesteps per episode (counted from each step() call)
        eval_env._max_episode_steps = max_num_timesteps

        # Keep track of object coordinates
        obj_coords = eval_env.get_obj_coords()
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

        # Beginning of episode time steps, done is max timesteps or lift reward achieved
        while not done:
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
                [ready_for_lift, _] = naive_check_grasp(f_dist_old, f_dist_new)

            #####
            # Not ready for lift, continue agent grasping following the policy
            if not ready_for_lift:
                action = policy.select_action(np.array(state[0:82]))
                next_state, reward, done, info = eval_env.step(action)
                cumulative_reward += reward
                avg_reward += reward
                curr_reward = reward

                # Cumulative reward
                ep_total_reward += reward
                ep_finger_reward += info["finger_reward"]
                ep_grasp_reward += info["grasp_reward"]
                ep_lift_reward += info["lift_reward"]

            else:  # Make it happen in one time step
                next_state, reward, done, info,  cumulative_reward = eval_lift_hand(eval_env, cumulative_reward,
                                                                                 curr_reward)
                if done:
                    avg_reward += reward

                    # Cumulative reward
                    ep_total_reward += reward
                    ep_finger_reward += info["finger_reward"]
                    ep_grasp_reward += info["grasp_reward"]
                    ep_lift_reward += info["lift_reward"]
                check_for_lift = False

            #####
            if reward > 25:
                success=1

            state = next_state
            prev_state_lift_check = curr_state_lift_check
            curr_state_lift_check = state

        # End of episode, record findings
        all_ep_reward_values["total_reward"].append(ep_total_reward)
        all_ep_reward_values["finger_reward"].append(ep_finger_reward)
        all_ep_reward_values["grasp_reward"].append(ep_grasp_reward)
        all_ep_reward_values["lift_reward"].append(ep_lift_reward)

        num_success+=success

        # Save initial object coordinate as success/failure
        x_val = (obj_coords[0])
        y_val = (obj_coords[1])
        x_val = np.asarray(x_val).reshape(1)
        y_val = np.asarray(y_val).reshape(1)

        if(success):
            seval_obj_posx = np.append(seval_obj_posx,x_val)
            seval_obj_posy = np.append(seval_obj_posy,y_val)
        else:
            feval_obj_posx = np.append(feval_obj_posx,x_val)
            feval_obj_posy = np.append(feval_obj_posy,y_val)

        total_evalx = np.append(total_evalx, x_val)
        total_evaly = np.append(total_evaly, y_val)

    avg_reward /= eval_episodes

    # Final average reward values over all episodes
    avg_rewards = {}
    avg_rewards["total_reward"] = np.average(all_ep_reward_values["total_reward"])
    avg_rewards["finger_reward"] = np.average(all_ep_reward_values["finger_reward"])
    avg_rewards["grasp_reward"] = np.average(all_ep_reward_values["grasp_reward"])
    avg_rewards["lift_reward"] = np.average(all_ep_reward_values["lift_reward"])

    print("---------------------------------------")
    print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
    print("---------------------------------------")

    ret = {"avg_reward": avg_reward, "avg_rewards": avg_rewards, "all_ep_reward_values": all_ep_reward_values, "num_success":num_success, "seval_obj_posx":seval_obj_posx, "seval_obj_posy":seval_obj_posy, "feval_obj_posx":feval_obj_posx, "feval_obj_posy":feval_obj_posy, "total_evalx":total_evalx,
           "total_evaly":total_evaly}
    return ret


def lift_hand(env_lift, tot_reward):
    """ Lift hand with set action velocities
    env_lift: Mujoco environment
    tot_reward: Total cumulative reward
    """
    action = np.array([wrist_lift_velocity, finger_lift_velocity, finger_lift_velocity,
                       finger_lift_velocity])
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
    action = np.array([wrist_lift_velocity, finger_lift_velocity, finger_lift_velocity,
                       finger_lift_velocity])
    next_state, reward, done, info = env_lift.step(action)
    if done:
        tot_reward = tot_reward - curr_reward + reward

    return next_state, reward, done, info, tot_reward

def write_tensor_plot(writer,episode_num,avg_reward,avg_rewards,actor_loss,critic_loss,critic_L1loss,critic_LNloss):
    """ Write important data about policy performance to tensorboard Summary Writer
    writer: Tensorboard Summary Writer
    episode_num: Current episode number
    avg_reward: Reward average over evaulation grasp trials
    avg_rewards: Dictionary of average reward values for Finger, Grasp, and Lift rewards over evaluation grasp trials
    actor_loss,critic_loss,critic_L1loss,critic_LNloss
    """
    writer.add_scalar("Episode total reward, Avg. " + str(args.eval_freq) + " episodes", avg_reward, episode_num)
    writer.add_scalar("NEW Episode total reward, Avg. " + str(args.eval_freq) + " episodes", avg_rewards["total_reward"], episode_num)
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
    return writer


def update_policy(evaluations, episode_num, num_episodes, writer, prob,
                  type_of_training, s_obj_posx, s_obj_posy, f_obj_posx, f_obj_posy, max_num_timesteps=30):
    """ Update policy network based on expert or agent step, evaluate every eval_freq episodes
    evaluations: Output average reward list for plotting
    episode_num: Current episode
    num_episodes: Max number of episodes to update over
    writer: Tensorboard writer (avg. reward, loss, etc.)
    prob: Probability of sampling from expert replay buffer
    type_of_training: Based on training mode ("pre-train", "eval", "test", etc.)
    s_obj_posx: Successful initial object x-axis coordinate
    s_obj_posy: Successful initial object y-axis coordinate
    f_obj_posx: Failed initial object x-axis coordinate
    f_obj_posy: Failed initial object y-axis coordinate
    max_num_timesteps: Maximum number of time steps within a RL episode
    """
    # Initialize OU noise, added to action output from policy
    noise = OUNoise(4)
    noise.reset()
    expl_noise = OUNoise(4, sigma=0.001)
    expl_noise.reset()

    # Holds heatmap coordinates for eval plots
    seval_obj_posx = np.array([])   # Successful evaluation object coords
    seval_obj_posy = np.array([])
    feval_obj_posx = np.array([])   # Failed evaluation object coords
    feval_obj_posy = np.array([])
    total_evalx = np.array([])      # Total evaluation object coords
    total_evaly = np.array([])

    # Stores reward boxplot data, Average reward per evaluation episodes
    finger_reward = [[]]
    grasp_reward = [[]]
    lift_reward = [[]]
    total_reward = [[]]

    for _ in range(num_episodes):
        env = gym.make(args.env_name)

        # Max number of timesteps to match the expert replay grasp trials
        env._max_episode_steps = max_num_timesteps

        # Fill training object list using latin square
        if env.check_obj_file_empty("objects.csv"):
            env.Generate_Latin_Square(args.max_episode, "objects.csv", shape_keys=requested_shapes)
        state, done = env.reset(env_name="env", shape_keys=requested_shapes, hand_orientation=requested_orientation,
                                mode=args.mode), False

        prev_state_lift_check = None
        curr_state_lift_check = state

        noise.reset()
        expl_noise.reset()
        episode_reward = 0
        obj_coords = env.get_obj_coords()
        replay_buffer.add_episode(1)
        timestep = 0  # Timestep counter is only used for testing purposes
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
                [ready_for_lift, _] = naive_check_grasp(f_dist_old, f_dist_new)

            # Follow policy until ready for lifting, then switch to set controller
            if not ready_for_lift:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                # Perform action obs, total_reward, done, info
                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state[0:82], action, next_state[0:82], reward, float(done))
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

        # Heatmap position data, get starting object position
        x_val = obj_coords[0]
        y_val = obj_coords[1]
        x_val = np.asarray(x_val).reshape(1)
        y_val = np.asarray(y_val).reshape(1)
        if lift_success is True:
            s_obj_posx = np.append(s_obj_posx, x_val)
            s_obj_posy = np.append(s_obj_posy, y_val)
        else:
            f_obj_posx = np.append(f_obj_posx, x_val)
            f_obj_posy = np.append(f_obj_posy, y_val)

        # Train agent after collecting sufficient data:
        if episode_num > 10:
            for learning in range(100):
                if args.batch_size is 0:
                    #print("SINGLE EP TRAIN: n-step is 5, doing single episode full trajectory")
                    # Single episode training using full trajectory
                    actor_loss, critic_loss, critic_L1loss, critic_LNloss = policy.train(env._max_episode_steps,
                                                                                     expert_replay_buffer,
                                                                                     replay_buffer, prob)
                else:
                    #print("BATCH TRAIN: n-step is 5, batch_size: ",args.batch_size," expert prob: ",prob)
                    # Batch training using n-steps
                    actor_loss, critic_loss, critic_L1loss, critic_LNloss = policy.train_batch(env._max_episode_steps,
                                                                                         expert_replay_buffer,
                                                                                         replay_buffer, prob)

        # Evaluation and recording data for tensorboard
        if episode_num > 10 and (episode_num) % args.eval_freq == 0: # episode_num + 1
            print("EVALUATING EPISODE AT: ",episode_num)
            eval_episodes = 100
            print("Evaluating with "+str(eval_episodes)+" grasping trials")
            eval_ret = eval_policy(policy, args.env_name, args.seed, requested_shapes, requested_orientation,
                                   mode=args.mode, eval_episodes=100)  # , compare=True)
            # Heatmap data - object starting coordinates for evaluation
            seval_obj_posx = np.append(seval_obj_posx, eval_ret["seval_obj_posx"])
            seval_obj_posy = np.append(seval_obj_posy, eval_ret["seval_obj_posy"])
            feval_obj_posx = np.append(feval_obj_posx, eval_ret["feval_obj_posx"])
            feval_obj_posy = np.append(feval_obj_posy, eval_ret["feval_obj_posy"])
            total_evalx = np.append(total_evalx, eval_ret["total_evalx"])
            total_evaly = np.append(total_evaly, eval_ret["total_evaly"])

            # Average rewards for finger, grasp, lift, total over evaluation episodes
            # final_avg_rewards = eval_ret["avg_rewards"]

            # Cumulative (over timesteps) reward data from each evaluation episode for boxplot
            all_ep_reward_values = eval_ret["all_ep_reward_values"]

            #print("BACK IN MAIN LOOP, len(all_ep_reward_values[total_reward]): ",len(all_ep_reward_values["total_reward"]))

            # Plot tensorboard metrics for learning analysis (average reward, loss, etc.)
            writer = write_tensor_plot(writer,episode_num,eval_ret["avg_reward"],eval_ret["avg_rewards"],actor_loss,critic_loss,critic_L1loss,critic_LNloss)

            # Insert boxplot code reference
            finger_reward[-1].append(all_ep_reward_values["finger_reward"])
            grasp_reward[-1].append(all_ep_reward_values["grasp_reward"])
            lift_reward[-1].append(all_ep_reward_values["lift_reward"])
            total_reward[-1].append(all_ep_reward_values["total_reward"])

            #print("BACK IN MAIN LOOP, np.asarray(finger_reward).shape: ", np.asarray(finger_reward).shape)

        # Evaluation plotting data directory
        evplot_saving_dir = "./eval_plots"
        if not os.path.isdir(evplot_saving_dir):
            os.mkdir(evplot_saving_dir)

        # Directory for x,y coordinate heatmap data
        heatmap_saving_dir = os.path.join(evplot_saving_dir, "heatmap/")
        if not os.path.isdir(heatmap_saving_dir):
            os.mkdir(heatmap_saving_dir)

        # Directory for boxplot reward data
        boxplot_saving_dir = os.path.join(evplot_saving_dir, "boxplot/")
        if not os.path.isdir(boxplot_saving_dir):
            os.mkdir(boxplot_saving_dir)

        # Save coordinates every 1000 episodes
        if episode_num > 10 and (episode_num) % 1000 == 0:
            print("Saving heatmap data at: ", heatmap_saving_dir)
            save_coordinates(seval_obj_posx, seval_obj_posy,
                             heatmap_saving_dir + "/eval_success", episode_num)
            save_coordinates(feval_obj_posx, feval_obj_posy,
                             heatmap_saving_dir + "/eval_fail", episode_num)
            save_coordinates(total_evalx, total_evaly, heatmap_saving_dir + "/eval_total", episode_num)
            seval_obj_posx = np.array([])
            seval_obj_posy = np.array([])
            feval_obj_posx = np.array([])
            feval_obj_posy = np.array([])
            total_evalx = np.array([])
            total_evaly = np.array([])

            print("Saving boxplot data at: ",boxplot_saving_dir)
            np.save(boxplot_saving_dir + "/finger_reward_" + str(episode_num),finger_reward)
            np.save(boxplot_saving_dir + "/grasp_reward_" + str(episode_num),grasp_reward)
            np.save(boxplot_saving_dir + "/lift_reward_" + str(episode_num),lift_reward)
            np.save(boxplot_saving_dir + "/total_reward_" + str(episode_num),total_reward)

            finger_reward = [[]]
            grasp_reward = [[]]
            lift_reward = [[]]
            total_reward = [[]]

        episode_num += 1

    return evaluations, episode_num, s_obj_posx, s_obj_posy, f_obj_posx, f_obj_posy


def pretrain_policy(tot_episodes,prob_exp):
    """ Pre-train the policy over a number of episodes, sampling from experience
    tot_episodes: Total number of episodes to update policy over
    prob_exp: Probability of sampling from expert replay buffer within training
    """
    print("---- Pretraining ----")
    # Tensorboard writer for tracking loss and average reward
    pre_writer = SummaryWriter(logdir="./kinova_gripper_strategy/{}_{}/".format("pretrain_" + args.policy_name, args.tensorboardindex))

    # Location of stored model
    pretrain_model_path = saving_dir + "/pre_DDPGfD_kinovaGrip_{}".format(datetime.datetime.now().strftime("%m_%d_%y_%H%M"))

    # Save the x,y coordinates for object starting position (success vs failed grasp and lift)
    pretrain_saving_dir = "./pretrain_plots"
    if not os.path.isdir(pretrain_saving_dir):
        os.mkdir(pretrain_saving_dir)

    # Stores average reward of pre-trained policy from evaluations list
    if not os.path.exists("./pretrain_results"):
        os.makedirs("./pretrain_results")
    evals = []
    curr_episode = 0         # Counts number of episodes done within training

    # Heatmap initial object coordinates for training
    s_pretrain_obj_posx = np.array([])  # Successful initial object coordinates: training
    s_pretrain_obj_posy = np.array([])
    f_pretrain_obj_posx = np.array([])  # Failed initial object coordinates: training
    f_pretrain_obj_posy = np.array([])

    # Update the policy based on experience replay
    evals, curr_episode, s_pretrain_obj_posx,  s_pretrain_obj_posy, f_pretrain_obj_posx,  f_pretrain_obj_posy = \
        update_policy(evals, curr_episode, tot_episodes, pre_writer, prob_exp, "PRE", s_pretrain_obj_posx,
                      s_pretrain_obj_posy, f_pretrain_obj_posx, f_pretrain_obj_posy)

    # Total initial object coordinates
    train_totalx = np.append(s_pretrain_obj_posx, f_pretrain_obj_posx)
    train_totaly = np.append(s_pretrain_obj_posy, f_pretrain_obj_posy)

    # Save object positions from training
    print("Success train num: ",len(s_pretrain_obj_posx))
    print("Fail train num: ", len(f_pretrain_obj_posx))
    print("Total train num: ", len(train_totalx))

    # Directory for x,y coordinate heatmap data
    heatmap_saving_dir = os.path.join(pretrain_saving_dir, "heatmap/")
    if not os.path.isdir(heatmap_saving_dir):
        os.mkdir(heatmap_saving_dir)
    save_coordinates(s_pretrain_obj_posx,s_pretrain_obj_posy,heatmap_saving_dir+"/train_success","")
    save_coordinates(f_pretrain_obj_posx,f_pretrain_obj_posy,heatmap_saving_dir+"/train_fail","")
    save_coordinates(train_totalx,train_totaly,heatmap_saving_dir+"/train_total","")

    print("Saving into {}".format(pretrain_model_path))
    policy.save(pretrain_model_path)

    return pretrain_model_path


def train_policy(tot_episodes,tr_prob):
    """ Train the policy over a number of episodes, sampling from experience
    tot_episodes: Total number of episodes to update policy over
    tr_prob: Probability of sampling from expert replay buffer within training
    """
    print("---- Training ----")
    # Setup training output location
    trplot_saving_dir = "./train_plots"
    if not os.path.isdir(trplot_saving_dir):
       os.mkdir(trplot_saving_dir)

    # Stores average reward of policy from evaluations list
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Initial model output setup
    model_save_path = saving_dir + "/DDPGfD_kinovaGrip_{}".format(datetime.datetime.now().strftime("%m_%d_%y_%H%M"))
    tr_writer = SummaryWriter(logdir="./kinova_gripper_strategy/{}_{}/".format(args.policy_name, args.tensorboardindex))
    tr_ep = int(tot_episodes/4)
    evals = []
    curr_episode = 0         # Counts number of episodes done within training
    red_expert_prob= 0.1

    # Heatmap initial object coordinates for training
    s_train_obj_posx = np.array([])  # Successful initial object coordinates: training
    s_train_obj_posy = np.array([])
    f_train_obj_posx = np.array([])  # Failed initial object coordinates: training
    f_train_obj_posy = np.array([])

    # Begin training updates
    for i in range(tot_episodes):
        evals, curr_episode, s_train_obj_posx,  s_train_obj_posy, f_train_obj_posx,  f_train_obj_posy =\
            update_policy(evals, curr_episode, tr_ep, tr_writer, tr_prob, "TRAIN", s_train_obj_posx, s_train_obj_posy,
                          f_train_obj_posx, f_train_obj_posy)
        tr_prob = tr_prob - red_expert_prob

    train_totalx = np.append(s_train_obj_posx, f_train_obj_posx)
    train_totaly = np.append(s_train_obj_posy, f_train_obj_posy)

    # Save object postions from training
    print("Success train num: ",len(s_train_obj_posx))
    print("Fail train num: ", len(f_train_obj_posx))
    print("Total train num: ", len(train_totalx))
    # Directory for x,y coordinate heatmap data
    heatmap_saving_dir = os.path.join(trplot_saving_dir, "heatmap/")
    if not os.path.isdir(heatmap_saving_dir):
        os.mkdir(heatmap_saving_dir)
    save_coordinates(s_train_obj_posx,s_train_obj_posy, heatmap_saving_dir+"/train_success", "")
    save_coordinates(f_train_obj_posx,f_train_obj_posy, heatmap_saving_dir+"/train_fail", "")
    save_coordinates(train_totalx,train_totaly, heatmap_saving_dir+"/train_total", "")

    print("Saving into {}".format(model_save_path))
    policy.save(model_save_path)

    return model_save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="DDPGfD")				# Policy name
    parser.add_argument("--env_name", default="gym_kinova_gripper:kinovagripper-v0") # OpenAI gym environment name
    parser.add_argument("--seed", default=2, type=int)					# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=100, type=int)		# How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=200, type=float)			# How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)		# Max time steps to run environment for
    parser.add_argument("--max_episode", default=20000, type=int)		# Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=0, type=int)			# Batch size for both actor and critic - Change to be 64
    parser.add_argument("--discount", default=0.995, type=float)			# Discount factor
    parser.add_argument("--tau", default=0.0005, type=float)				# Target network update rate
    parser.add_argument("--policy_noise", default=0.01, type=float)		# Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.05, type=float)		# Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
    parser.add_argument("--tensorboardindex", type=str, default=None)	# Tensorboard log name, found in kinova_gripper_strategy/
    parser.add_argument("--expert_replay_size", default=20000, type=int)	# Number of episode for loading expert trajectories
    parser.add_argument("--saving_dir", type=str, default=None)         # Directory name to save policy within policies/
    parser.add_argument("--shapes", default='CubeS', action='store', type=str) # Requested shapes to use (in format of object keys)
    parser.add_argument("--hand_orientation", action='store', type=str)         # Requested shapes to use (in format of object keys)
    parser.add_argument("--mode", action='store', type=str, default="train")    # Mode to run experiments with: (naive_only, expert_only, expert, pre-train, train, rand_train, test)
    parser.add_argument("--agent_replay_size", default=10100, type=int)         # Maximum size of agent's replay buffer
    parser.add_argument("--expert_prob", default=1, type=float)           # Probability of sampling from expert replay buffer (opposed to agent replay buffer)

    args = parser.parse_args()

    """ Setup the environment, state, and action space """

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: "+file_name)
    print("---------------------------------------")

    # Make initial environment
    env = gym.make(args.env_name)

    # Set seeds for randomization
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set dimensions for state and action spaces - policy initialization
    state_dim = 82  # State dimension dependent on the length of the state space
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_action_trained = env.action_space.high  # a vector of max actions
    n = 5   # n step look ahead for the policy
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

    # Fill pre-training object list using latin square method
    env.Generate_Latin_Square(args.max_episode,"objects.csv", shape_keys=requested_shapes)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "n": n,
        "discount": args.discount,
        "tau": args.tau,
        "batch_size": args.batch_size
    }

    ''' Initialize policy '''
    if args.policy_name == "DDPGfD":
        policy = DDPGfD.DDPGfD(**kwargs)
    else:
        print("No such policy")
        raise ValueError

    if args.saving_dir is None:
        args.saving_dir = "%s_%s" % (args.policy_name, args.mode)

    if args.tensorboardindex is None:
        args.tensorboardindex = "%s_%s" % (args.policy_name, args.mode)

    # Print variables set based on command line input
    print("Saving dir: ",args.saving_dir)
    print("Tensorboard index: ",args.tensorboardindex)
    print("Policy: ", args.policy_name)
    print("Requested_shapes: ",requested_shapes)
    print("Requested Hand orientation: ", requested_orientation)
    print("Batch Size: ", args.batch_size)
    print("Expert Sampling Probability: ", args.expert_prob)
    print("Generating " + str(args.max_episode) + " episodes!")
    print("---------------------------------------")

    ''' Select replay buffer type
    # Replay buffer that can use multiple n-steps
    #replay_buffer = utils.ReplayBuffer_VarStepsEpisode(state_dim, action_dim, expert_replay_size)
    #replay_buffer = utils.ReplayBuffer_NStep(state_dim, action_dim, expert_replay_size)

    # Replay buffer that samples only one episode
    #replay_buffer = utils.ReplayBuffer_episode(state_dim, action_dim, env._max_episode_steps, args.expert_replay_size, args.max_episode)
    #replay_buffer = GenerateExpertPID_JointVel(args.expert_replay_size, replay_buffer)
    '''

    # Initialize Queue Replay Buffer: replay buffer manages its size like a queue, popping off the oldest episodes
    replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, agent_replay_size)

    # Default expert pid file path
    expert_file_path = "./expert_replay_data/Expert_data_01_20_21_0450/"

    # Default pre-trained policy file path
    pretrain_model_save_path = "./policies/pretrain_policy/exp_NO_graspclassifier_expert_CubeS_2_2_21/pre_DDPGfD_kinovaGrip_01_25_21_2334"

    # Create directory to hold trained policy
    saving_dir = "./policies/" + args.saving_dir
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)

    # Initialize timer to analyze run times
    total_time = Timer()
    total_time.start()

    # Determine replay buffer/policy function calls based on mode (expert, pre-train, train, rand_train, test)
    # Generate expert data based on Expert nudge controller only
    if args.mode == "expert_only":
        print("MODE: Expert ONLY")
        # Initialize expert replay buffer, then generate expert pid data to fill it
        expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
        expert_replay_buffer, expert_file_path = GenerateExpertPID_JointVel(expert_replay_size, expert_replay_buffer, pid_mode="expert_only")
        print("Expert ONLY expert_file_path: ",expert_file_path, "\n", expert_replay_buffer)

    # Generate expert data based on Naive controller only
    elif args.mode == "naive_only":
        print("MODE: Naive ONLY")
        # Initialize expert replay buffer, then generate expert pid data to fill it
        expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
        expert_replay_buffer, expert_file_path = GenerateExpertPID_JointVel(expert_replay_size, expert_replay_buffer, pid_mode="naive_only")
        print("Naive ONLY expert_file_path: ",expert_file_path, "\n", expert_replay_buffer)

    # Generate expert data based on interpolating naive and expert strategies
    elif args.mode == "expert":
        print("MODE: Expert (Interpolation)")
        # Initialize expert replay buffer, then generate expert pid data to fill it
        expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
        expert_replay_buffer, expert_file_path = GenerateExpertPID_JointVel(expert_replay_size, expert_replay_buffer, pid_mode="expert_naive")
        print("Expert (Interpolation) expert_file_path: ",expert_file_path, "\n", expert_replay_buffer)

    # Pre-train policy using expert data, save pre-trained policy for use in training
    elif args.mode == "pre-train":
        print("MODE: Pre-train")
        # Load expert data from saved expert pid controller replay buffer
        expert_replay_buffer = store_saved_data_into_replay(replay_buffer, expert_file_path)

        # Pre-train policy based on expert data
        pretrain_model_save_path = pretrain_policy(args.max_episode,args.expert_prob)
        print("pretrain_model_save_path: ",pretrain_model_save_path)

    # Train policy starting with pre-trained policy and sampling from experience
    elif args.mode == "train":
        print("MODE: Train (w/ pre-trained policy")
        # Load expert replay buffer and pre-trained policy
        expert_replay_buffer = store_saved_data_into_replay(replay_buffer, expert_file_path)

        # Load Pre-Trained policy
        policy.load(pretrain_model_save_path)
        print("Expert replay: ",expert_file_path)
        print("Pre-trained policy: ",pretrain_model_save_path)
        train_policy(args.max_episode,args.expert_prob)

    # Train policy given randomly initialized policy
    elif args.mode == "rand_train":
        print("MODE: Train (Random init policy)")
        if expert_replay_size is 0:
            expert_replay_buffer = None
        else:
            # Initialize expert replay buffer, then generate expert pid data to fill it
            expert_replay_buffer = utils.ReplayBuffer_Queue(state_dim, action_dim, expert_replay_size)
            expert_replay_buffer, expert_file_path = GenerateExpertPID_JointVel(expert_replay_size,
                                                                                expert_replay_buffer)
        train_policy(args.max_episode,args.expert_prob)

    # Test policy over certain number of episodes -- In Progress
    elif args.mode == "test":
        print("MODE: Test")

        # Load policy
        policy.load(pretrain_model_save_path)   # Change to be complete, trained policy
        print("Policy: ", pretrain_model_save_path)

        # Evaluate policy over certain number of episodes
        eval_ret = eval_policy(policy, args.env_name, args.seed, requested_shapes, requested_orientation,
                               mode=args.mode, eval_episodes=args.max_episode)
        # Add further evaluation here
    else:
        print("Invalid mode input")

    total_time.stop()


