
import numpy as np
import torch
import DDPGfD
import main_DDPGfD
import gym


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
	

# env = gym.make("HalfCheetah-v2")
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0] 
# max_action = float(env.action_space.high[0])
# policy = DDPGfD.DDPGfD(state_dim, action_dim, max_action)
# policy.load("DDPGfD_kinovaGrip_10_21_19_1801")
# main_DDPGfD.eval_policy(policy, "HalfCheetah-v2", 2)