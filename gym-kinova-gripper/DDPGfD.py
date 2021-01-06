import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
		self.l2 = nn.Linear(400, 300)
		torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
		self.l3 = nn.Linear(300, action_dim)
		torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

		self.max_action = max_action


	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.sigmoid(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
		self.l2 = nn.Linear(400, 300)
		torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
		self.l3 = nn.Linear(300, 1)
		torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], -1)))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DDPGfD(object):
	def __init__(self, state_dim, action_dim, max_action, n, discount=0.995, tau=0.0005):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-4)

		self.discount = discount
		self.tau = tau
		self.n = n
		self.network_repl_freq = 10
		self.total_it = 0

		# note: parameterize this later!!!
		self.batch_size = 64

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, episode_step, expert_replay_buffer, replay_buffer=None):
		self.total_it += 1

		# Sample replay buffer
		if replay_buffer is not None and expert_replay_buffer is None: # Only use agent replay
			expert_or_random = "agent"
			# print("REPLAY BUFFER IS NOT NONE")
		elif replay_buffer is None and expert_replay_buffer is not None: # Only use expert replay
			expert_or_random = "expert"
			# print("REPLAY BUFFER IS NONE")
		else:
			# print("PROBABILITY EXPERT OR AGENT")
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[0.8, 0.2])
			#expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[0.7, 0.3])

		if expert_or_random == "expert":
			# print("IN EXPERT BUFFER")
			state, action, next_state, reward, not_done = expert_replay_buffer.sample()
		else:
			# print("IN NORMAL BUFFER")
			state, action, next_state, reward, not_done = replay_buffer.sample()
		#state, action, next_state, reward, not_done = replay_buffer.sample_wo_expert()

		# new sampling procedure for n step rollback
		#state, action, next_state, reward, not_done = replay_buffer.sample_batch_nstep()

		# print("=======================state===================")
		# print(state.shape)
		# print("=======================next_state===================")
		# print(next_state.shape)
		#
		# print("=====================action====================")
		# print(action.shape)

		# Old implementation of target Q
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (self.discount * target_Q).detach()  # bellman equation

		''' Newly added N step code
		target_Q = self.critic_target(next_state[:, 0], self.actor_target(next_state[:, 0]))
		print("before regular")
		print(target_Q.shape)
		print(reward[:, 0].shape)
		target_Q = reward[:, 0] + (self.discount * target_Q).detach() #bellman equation
		'''

		# print("======================target_Q===================")
		# print(target_Q.shape)
		#
		# print("next state smaller")
		# print(next_state[(self.n - 1):].shape)
		#
		# print("number of episode steps")
		# print(episode_step)

		"""
		This is the updated version, assuming that we're sampling from a batch.
		"""
		''' Newly added N step code
		target_action = self.actor_target(next_state[:, -1])
		target_critic_val = self.critic_target(next_state[:, -1], target_action)  # shape: (self.batch_size, 1)

		n_step_return = torch.zeros(self.batch_size).to(device)  # shape: (self.batch_size,)
		# note: might need to pass n properly from higher state!!!

		print("================================reward shape=======================")
		print(reward[:, 0].shape)  # idk the shape here, please record

		for i in range(self.n):
			n_step_return += (self.discount ** i) * reward[:, i].squeeze(-1)

		print("====================n step return shape====================")
		print(n_step_return.shape)

		# this is the n step return with the added value fn estimation
		target_QN = n_step_return + (self.discount ** self.n) * target_critic_val.squeeze(-1)
		target_QN = target_QN.unsqueeze(dim=-1)

		print("=======================target QN")
		print(target_QN.shape)
		'''

		# Old version: Compute the target Q_N value
		rollreward = []
		target_QN = self.critic_target(next_state[(self.n - 1):], self.actor_target(next_state[(self.n - 1):]))

		'''
		# Checks episode size versus value of n (In case n is larger than the number of timesteps)
		if state.shape[0] < 3:
			for i in range(state.shape[0]):
				roll_reward = (self.discount) * reward[i].item()
				rollreward.append(roll_reward)
		else:
		'''
		# print("state.shape[0]: ", state.shape[0])
		# print("episode_step: ",episode_step)
		ep_timesteps = episode_step
		if state.shape[0] < episode_step:
			ep_timesteps = state.shape[0]

		for i in range(ep_timesteps):
			if i >= (self.n - 1):
				roll_reward = (self.discount**(self.n - 1)) * reward[i].item() + (self.discount**(self.n - 2)) * reward[i - (self.n - 2)].item() + (self.discount ** 0) * reward[i-(self.n - 1)].item()
				rollreward.append(roll_reward)


		if len(rollreward) != ep_timesteps - (self.n - 1):
			raise ValueError

		#print("roll reward before reshape: ")
		#print(rollreward)
		#print(len(rollreward))

		rollreward = torch.FloatTensor(np.array(rollreward).reshape(-1,1)).to(device)
		# print("rollreward.get_shape(): ", rollreward.size())
		# print("target_QN.get_shape(): ", target_QN.size())
		# print("self.discount: ", self.discount)
		# print("self.n.: ", self.n)

		# print("================SHAPE DUMP=============")
		# print(rollreward.shape)
		# print(((self.discount ** self.n) * target_QN).shape)

		# Old code: calculate target network
		target_QN = rollreward + (self.discount ** self.n) * target_QN #bellman equation <= this is the final N step return

		# Old code: Get current Q estimate
		current_Q = self.critic(state, action)

		# New implementation
		#current_Q = self.critic(state[:, 0], action[:, 0])

		# Old code: Get current Q estimate for n-step return
		#current_Q_n = self.critic(state[:(episode_step - (self.n - 1))], action[:(episode_step - (self.n - 1))])
		current_Q_n = self.critic(state[:(ep_timesteps - (self.n - 1))], action[:(ep_timesteps - (self.n - 1))])

		# New Updated for new rollback method
		#current_Q_n = self.critic(state[:, -1], action[:, -1])

		# print("======================Q shapes finallly==============")
		# print(current_Q.shape)
		# print(target_Q.shape)
		# print(current_Q_n.shape)
		# print(target_QN.shape)
		# print("==============end printing pain==================")

		# L_1 loss (Loss between current state, action and reward, next state, action)
		critic_L1loss = F.mse_loss(current_Q, target_Q)

		# L_2 loss (Loss between current state, action and reward, n state, n action)
		critic_LNloss = F.mse_loss(current_Q_n, target_QN)

		# Total critic loss
		lambda_1 = 0.5 # hyperparameter to control n loss
		critic_loss = critic_L1loss + lambda_1 * critic_LNloss

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()

		# Optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if self.total_it % self.network_repl_freq == 0:
			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return actor_loss.item(), critic_loss.item(), critic_L1loss.item(), critic_LNloss.item()

	def train_batch(self, replay_buffer, episode_step):
		self.total_it += 1

		# new sampling procedure for n step rollback
		state, action, next_state, reward, not_done = replay_buffer.sample_batch_nstep()

		# print("=======================state===================")
		# print(state.shape)
		# print("=======================next_state===================")
		# print(next_state.shape)
		#
		# print("=====================action====================")
		# print(action.shape)

		# print("==========modified states and actions==============")
		# print(state[:, -1])
		# print(action[:, -1])



		# episode_step = len(state) # for varying episode sets


		target_Q = self.critic_target(next_state[:, 0], self.actor_target(next_state[:, 0]))
		# print("before regular")
		# print(target_Q.shape)
		# print(reward[:, 0].shape)
		target_Q = reward[:, 0] + (self.discount * target_Q).detach() #bellman equation

		# print("======================target_Q===================")
		# print(target_Q.shape)
		#
		# print("next state smaller")
		# print(next_state[(self.n - 1):].shape)
		#
		# print("number of episode steps")
		# print(episode_step)

		"""
		This is the updated version, assuming that we're sampling from a batch.
		"""
		target_action = self.actor_target(next_state[:, -1])
		target_critic_val = self.critic_target(next_state[:, -1], target_action)  # shape: (self.batch_size, 1)

		n_step_return = torch.zeros(self.batch_size).to(device)  # shape: (self.batch_size,)
		# note: might need to pass n properly from higher state!!!

		# print("================================reward shape=======================")
		# print(reward[:, 0].shape)  # idk the shape here, please record

		for i in range(self.n):
			n_step_return += (self.discount ** i) * reward[:, i].squeeze(-1)

		# print("====================n step return shape====================")
		# print(n_step_return.shape)

		# this is the n step return with the added value fn estimation
		target_QN = n_step_return + (self.discount ** self.n) * target_critic_val.squeeze(-1)
		target_QN = target_QN.unsqueeze(dim=-1)

		# print("=======================target QN")
		# print(target_QN.shape)

		# New implementation
		current_Q = self.critic(state[:, 0], action[:, 0])

		# New Updated for new rollback method
		current_Q_n = self.critic(state[:, -1], action[:, -1])

		# print("======================Q shapes finallly==============")
		# print(current_Q.shape)
		# print(target_Q.shape)
		# print(current_Q_n.shape)
		# print(target_QN.shape)
		# print("==============end printing pain==================")

		# L_1 loss (Loss between current state, action and reward, next state, action)
		critic_L1loss = F.mse_loss(current_Q, target_Q)

		# L_2 loss (Loss between current state, action and reward, n state, n action)
		critic_LNloss = F.mse_loss(current_Q_n, target_QN)

		# Total critic loss
		lambda_1 = 0.5 # hyperparameter to control n loss
		critic_loss = critic_L1loss + lambda_1 * critic_LNloss

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()

		# Optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if self.total_it % self.network_repl_freq == 0:
			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return actor_loss.item(), critic_loss.item(), critic_L1loss.item(), critic_LNloss.item()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))