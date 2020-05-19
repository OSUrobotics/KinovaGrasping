
import numpy as np
import torch
import DDPGfD
import main_DDPGfD
import gym
import pandas as pd
import matplotlib.pyplot as plt

def pretrain_from_agent(policy, env, replay_buffer, episode_num):
	policy.load("DDPGfD_kinovaGrip_10_21_19_1801")

	for _ in range(episode_num):
		state, done = env.reset(), False

		while not done:
			action = policy.select_action(np.array(state))
			next_state, reward, done, _ = env.step(action)
			replay_buffer.add(state, action, next_state, reward, done)
			state = next_state

	return replay_buffer
	

env = gym.make("gym_kinova_gripper:kinovagripper-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])
policy = DDPGfD.DDPGfD(state_dim, action_dim, max_action)
# policy.load("./policies/new_expert_action/DDPGfD_kinovaGrip_01_06_20_0819")

filepath="./policies/batch_policies/"
addons=["DDPGfD_kinovaGrip_10_22_19_2151","exp1s2_graspclassifier_v6/DDPGfD_kinovaGrip_01_27_20_1955","exp2s2_w_graspclassifier_v5/DDPGfD_kinovaGrip_01_27_20_1838","exp3s1_graspclassifier_v5/DDPGfD_kinovaGrip_01_27_20_1135","exp1s2_wo_graspclassifier/DDPGfD_kinovaGrip_01_13_20_1616","exp2s2_wo_graspclassifier/DDPGfD_kinovaGrip_01_15_20_0917","all_objs_wo_graspclassifier/DDPGfD_kinovaGrip_01_08_20_1102","exp1s2_loc_graspclassifier_v6a2/DDPGfD_kinovaGrip_01_29_20_1142","exp2s2_w_Locgraspclassifier/DDPGfD_kinovaGrip_01_29_20_1237","exp3s1_loc_graspclassifier_v5/DDPGfD_kinovaGrip_01_28_20_1049","exp3_wo_graspclassifier_local3/DDPGfD_kinovaGrip_01_31_20_0123"]
tot_rewards=np.zeros([7,40],dtype=bool)
used_policies=[]
for i in range(7):
    #policy.load("./policies/exp1s2_graspclassifier_v6/DDPGfD_kinovaGrip_01_27_20_1955")
    policy.load(filepath+addons[i])
    print(addons[i])
    used_policies.append(addons[i])
    if i==0:
        policy.load(filepath+addons[0])
        print(addons[0])
        _ ,tot_rewards[i]= main_DDPGfD.eval_policy(policy, "gym_kinova_gripper:kinovagripper-v0", 2,eval_episodes=40)
        #print('calculated avg rewards',np.average(tot_rewards[i]))
        #print('calculated std dev', np.std(tot_rewards[i]))
    
    else:
        _ ,tot_rewards[i]= main_DDPGfD.eval_policy(policy, "gym_kinova_gripper:kinovagripper-v0", 2,eval_episodes=40)
        #print('calculated avg rewards',np.average(tot_rewards[i]))
        #print('calculated std dev', np.std(tot_rewards[i]))
fig, ax = plt.subplots()
x = np.arange(len(tot_rewards))  # the label locations
important_stuff=tot_rewards[:,0:3]
important_stuff=np.append(important_stuff,tot_rewards[:,20:26],axis=1)
important_stuff=np.append(important_stuff,tot_rewards[:,37:],axis=1)
#print(important_stuff)
x = np.arange(len(important_stuff))
box_trues=np.sum(tot_rewards[:,0:23],axis=1)
cylinder_trues=np.sum(tot_rewards[:,23:],axis=1)
print('Policies used, ', used_policies)
print('Number of Box successes (out of 23), ', box_trues)
print('Number of Cylinder successes (out of 17), ',cylinder_trues)
box_trues=np.sum(important_stuff[:,0:6],axis=1)
cylinder_trues=np.sum(important_stuff[:,6:],axis=1)
print('Edge case Box successes (out of 6), ',box_trues)
print('Edge case Cylinder successes (out of 6), ',cylinder_trues)
width = 0.35  # the width of the bars

rects1 = ax.bar(x - width/2, np.sum(tot_rewards,axis=1), width, label='Success')
rects2 = ax.bar(x + width/2, np.sum(np.invert(tot_rewards),axis=1), width, label='Failure')
ax.set(xlim=(-0.35,6.35),ylim=(0, 42))
plt.legend()
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, np.sum(important_stuff,axis=1), width, label='Success')
rects2 = ax.bar(x + width/2, np.sum(np.invert(important_stuff),axis=1), width, label='Failure')
ax.set(xlim=(-0.35,6.35),ylim=(0, 10))
plt.legend()
plt.show()