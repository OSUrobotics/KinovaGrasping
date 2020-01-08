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
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.sigmoid(self.l3(a)) 


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DDPGfD(object):
	def __init__(self, state_dim, action_dim, max_action, n=5, discount=0.995, tau=0.0005):
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

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, episode_step):
		self.total_it += 1

		# Sample replay buffer 
		# state, action, next_state, reward, not_done = replay_buffer.sample()
		state, action, next_state, reward, not_done = replay_buffer.sample_wo_expert()

		
		# episode_step = len(state) # for varying episode sets

		# pdb.set_trace()
		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (self.discount * target_Q).detach() #bellman equation

		# Compute the target Q_N value
		rollreward = []
		target_QN = self.critic_target(next_state[(self.n - 1):], self.actor_target(next_state[(self.n - 1):]))
		for i in range(episode_step):
			if i >= (self.n - 1):
				roll_reward = (self.discount**(self.n - 1)) * reward[i].item() + (self.discount**(self.n - 2)) * reward[i - (self.n - 2)].item() + (self.discount ** 0) * reward[i-(self.n - 1)].item()
				rollreward.append(roll_reward)

		if len(rollreward) != episode_step - (self.n - 1):
			raise ValueError

		rollreward = torch.FloatTensor(np.array(rollreward).reshape(-1,1)).to(device)
		target_QN = rollreward + (self.discount ** self.n) * target_QN #bellman equation

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Get current Q estimate for n-step return 
		current_Q_n = self.critic(state[:(episode_step - (self.n - 1))], action[:(episode_step - (self.n - 1))])

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