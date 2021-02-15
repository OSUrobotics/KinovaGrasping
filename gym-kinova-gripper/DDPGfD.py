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
	def __init__(self, state_dim, action_dim, max_action, n, discount=0.995, tau=0.0005, batch_size=64):
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

		self.batch_size = batch_size

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, episode_step, expert_replay_buffer, replay_buffer=None, prob=0.7):
		""" Update policy based on full trajectory of one episode """
		self.total_it += 1

		# Determine which replay buffer to sample from
		if replay_buffer is not None and expert_replay_buffer is None: # Only use agent replay
			expert_or_random = "agent"
		elif replay_buffer is None and expert_replay_buffer is not None: # Only use expert replay
			expert_or_random = "expert"
		else:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[prob, round(1. - prob, 2)])

		if expert_or_random == "expert":
			state, action, next_state, reward, not_done = expert_replay_buffer.sample()
		else:
			state, action, next_state, reward, not_done = replay_buffer.sample()

		"""
		non_zero_count = 0
		for elem in reward:
			if elem != 0:
				non_zero_count += 1

		print("ORIGINAL TRAIN, reward non_zero count: ", non_zero_count)
		"""

		# Target Q network
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (self.discount * target_Q).detach()  # bellman equation

		# Compute the target Q_N value
		rollreward = []
		target_QN = self.critic_target(next_state[(self.n - 1):], self.actor_target(next_state[(self.n - 1):]))

		ep_timesteps = episode_step
		if state.shape[0] < episode_step:
			ep_timesteps = state.shape[0]

		for i in range(ep_timesteps):
			if i >= (self.n - 1):
				roll_reward = (self.discount**(self.n - 1)) * reward[i].item() + (self.discount**(self.n - 2)) * reward[i - (self.n - 2)].item() + (self.discount ** 0) * reward[i-(self.n - 1)].item()
				rollreward.append(roll_reward)

		if len(rollreward) != ep_timesteps - (self.n - 1):
			raise ValueError

		rollreward = torch.FloatTensor(np.array(rollreward).reshape(-1,1)).to(device)

		# Calculate target network
		target_QN = rollreward + (self.discount ** self.n) * target_QN #bellman equation <= this is the final N step return

		# Get current Q estimate
		current_Q = self.critic(state, action)

		current_Q_n = self.critic(state[:(ep_timesteps - (self.n - 1))], action[:(ep_timesteps - (self.n - 1))])

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

	"""
	def train_batch(self, episode_step, expert_replay_buffer, replay_buffer, prob=0.7):
		# Update policy networks based on batch_size of episodes using n-step returns 
		self.total_it += 1

		# Sample replay buffer
		if replay_buffer is not None and expert_replay_buffer is None: # Only use agent replay
			#print("AGENT")
			expert_or_random = "agent"
			state, action, next_state, reward, not_done = replay_buffer.sample_batch_nstep(self.batch_size,"AGENT")
		elif replay_buffer is None and expert_replay_buffer is not None: # Only use expert replay
			#print("EXPERT")
			expert_or_random = "expert"
			state, action, next_state, reward, not_done = expert_replay_buffer.sample_batch_nstep(self.batch_size,"EXPERT")
		else:
			#print("MIX OF AGENT AND EXPERT")
			# Get prob % of batch from expert and (1-prob) from agent
			agent_batch_size = int(self.batch_size * (1 - prob))
			expert_batch_size = self.batch_size - agent_batch_size
			# Get batches from respective replay buffers
			agent_state, agent_action, agent_next_state, agent_reward, agent_not_done = replay_buffer.sample_batch_nstep(agent_batch_size,"AGENT")
			expert_state, expert_action, expert_next_state, expert_reward, expert_not_done = expert_replay_buffer.sample_batch_nstep(expert_batch_size,"EXPERT")

			# Concatenate batches of agent and expert experience to get batch_size tensors of experience
			state = torch.cat((torch.squeeze(agent_state), torch.squeeze(expert_state)), 0)
			action = torch.cat((torch.squeeze(agent_action), torch.squeeze(expert_action)), 0)
			next_state = torch.cat((torch.squeeze(agent_next_state), torch.squeeze(expert_next_state)), 0)
			reward = torch.cat((torch.squeeze(agent_reward), torch.squeeze(expert_reward)), 0)
			not_done = torch.cat((torch.squeeze(agent_not_done), torch.squeeze(expert_not_done)), 0)
			if self.batch_size == 1:
				state = state.unsqueeze(0)
				action = action.unsqueeze(0)
				next_state = next_state.unsqueeze(0)
				reward = reward.unsqueeze(0)
				not_done = not_done.unsqueeze(0)

		
		# Add dimension to transpose values for multiplication to get Target Q
		reward = reward.unsqueeze(-1)
		not_done = not_done.unsqueeze(-1)

		# Target Q

		# Only using first timestep as that is the step we are currently sampling
		# (we only get nstep for roll reward calculation)
		target_Q = self.critic_target(next_state[:, 0], self.actor_target(next_state[:, 0]))
		# target_Q [64, 1] = critic_target(next_state: [64, 82], actor_target: [64, 4])
		# 64 batch, 1 q-value

		target_Q = reward[:, 0] + (self.discount * target_Q).detach() #bellman equation
		# [64, 1] = [64, 1] + 1 * [64, 1]

		print("new target_Q.size(): ", target_Q.size())

		print("\n Calculating Q_N")
		# Compute the target Q_N value
		rollreward = []
		print("next_state.size(): ",next_state.size())
		print("self.actor_target(next_state).size(): ",self.actor_target(next_state).size())
		target_QN = self.critic_target(next_state, self.actor_target(next_state))
		print("target_QN.size(): ",target_QN.size())
		# [64, 5, 1] = self.critic_target([64,5,82]=next_state, [64,5,4]=self.actor_target([64,5,82]))
		quit()

		ep_timesteps = episode_step
		if state.shape[0] < episode_step:
			ep_timesteps = state.shape[0]

		for i in range(ep_timesteps):
			if i >= (self.n - 1):
				roll_reward = (self.discount**(self.n - 1)) * reward[i].item() + (self.discount**(self.n - 2)) * reward[i - (self.n - 2)].item() + (self.discount ** 0) * reward[i-(self.n - 1)].item()
				rollreward.append(roll_reward)

		if len(rollreward) != ep_timesteps - (self.n - 1):
			raise ValueError

		rollreward = torch.FloatTensor(np.array(rollreward).reshape(-1,1)).to(device)

		###
		#target_action = self.actor_target(next_state[:, -1])
		#target_critic_val = self.critic_target(next_state[:, -1], target_action)  # shape: (self.batch_size, 1)

		#n_step_return = torch.zeros(self.batch_size).to(device)  # shape: (self.batch_size,)

		#for i in range(self.n):
		#	n_step_return += (self.discount ** i) * reward[:, i].squeeze(-1)

		# this is the n step return with the added value fn estimation
		#target_QN = n_step_return + (self.discount ** self.n) * target_critic_val.squeeze(-1)
		#target_QN = target_QN.unsqueeze(dim=-1)
		
		#####

		# Calculate target network
		target_QN = rollreward + (self.discount ** self.n) * target_QN #bellman equation <= this is the final N step return

		# Get current Q estimate
		current_Q = self.critic(state, action)

		current_Q_n = self.critic(state[:(ep_timesteps - (self.n - 1))], action[:(ep_timesteps - (self.n - 1))])

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
	"""


	# PREVIOUS Implementation STEPH + ADAM
	def train_batch(self, episode_step, expert_replay_buffer, replay_buffer, prob=0.7):
		# Update policy networks based on batch_size of episodes using n-step returns 
		self.total_it += 1

		# Sample replay buffer
		if replay_buffer is not None and expert_replay_buffer is None: # Only use agent replay
			#print("AGENT")
			expert_or_random = "agent"
			state, action, next_state, reward, not_done = replay_buffer.sample_batch_nstep(self.batch_size,"AGENT")
		elif replay_buffer is None and expert_replay_buffer is not None: # Only use expert replay
			#print("EXPERT")
			expert_or_random = "expert"
			state, action, next_state, reward, not_done = expert_replay_buffer.sample_batch_nstep(self.batch_size,"EXPERT")
		else:
			#print("MIX OF AGENT AND EXPERT")
			# Get prob % of batch from expert and (1-prob) from agent
			agent_batch_size = int(self.batch_size * (1 - prob))
			expert_batch_size = self.batch_size - agent_batch_size
			# Get batches from respective replay buffers
			#print("SAMPLING FROM AGENT...")
			agent_state, agent_action, agent_next_state, agent_reward, agent_not_done = replay_buffer.sample_batch_nstep(agent_batch_size,"AGENT")
			#print("SAMPLING FROM EXPERT...")
			expert_state, expert_action, expert_next_state, expert_reward, expert_not_done = expert_replay_buffer.sample_batch_nstep(expert_batch_size,"EXPERT")

			# Concatenate batches of agent and expert experience to get batch_size tensors of experience
			state = torch.cat((torch.squeeze(agent_state), torch.squeeze(expert_state)), 0)
			action = torch.cat((torch.squeeze(agent_action), torch.squeeze(expert_action)), 0)
			next_state = torch.cat((torch.squeeze(agent_next_state), torch.squeeze(expert_next_state)), 0)
			reward = torch.cat((torch.squeeze(agent_reward), torch.squeeze(expert_reward)), 0)
			not_done = torch.cat((torch.squeeze(agent_not_done), torch.squeeze(expert_not_done)), 0)
			if self.batch_size == 1:
				state = state.unsqueeze(0)
				action = action.unsqueeze(0)
				next_state = next_state.unsqueeze(0)
				reward = reward.unsqueeze(0)
				not_done = not_done.unsqueeze(0)

		reward = reward.unsqueeze(-1)
		not_done = not_done.unsqueeze(-1)

		"""
		non_zero_count = 0
		for row in reward:
			for elem in row:
				if elem != 0:
					non_zero_count += 1

		print("TRAIN BATCH, reward non_zero count: ", non_zero_count)
		if non_zero_count > 1:
			print("****FOUND A REWARD BIGGER THAN ZERO !!!!!")
			#quit()
		"""

		target_Q = self.critic_target(next_state[:, 0], self.actor_target(next_state[:, 0]))
		target_Q = reward[:, 0] + (self.discount * target_Q).detach() #bellman equation

		#This is the updated version, assuming that we're sampling from a batch.
		
		target_action = self.actor_target(next_state[:, -1])
		target_critic_val = self.critic_target(next_state[:, -1], target_action)  # shape: (self.batch_size, 1)

		n_step_return = torch.zeros(self.batch_size).to(device)  # shape: (self.batch_size,)

		for i in range(self.n):
			n_step_return += (self.discount ** i) * reward[:, i].squeeze(-1)

		# this is the n step return with the added value fn estimation
		target_QN = n_step_return + (self.discount ** self.n) * target_critic_val.squeeze(-1)
		target_QN = target_QN.unsqueeze(dim=-1)

		# New implementation
		current_Q = self.critic(state[:, 0], action[:, 0])
		#print("state[:, 0]: ",state[:, 0])
		#print("action[:, 0]: ", action[:, 0])
		#print("current_Q: ",current_Q)

		# New Updated for new rollback method
		current_Q_n = self.critic(state[:, -1], action[:, -1])

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