import numpy as np
import torch
import pdb
import random

class ReplayBuffer_Queue(object):
	def __init__(self, state_dim, action_dim, max_episode=10100, n_steps=5, batch_size=64):
		self.max_episode = max_episode		# Maximum number of episodes, limit to when we remove old episodes
		self.size = 0				# Full size of the replay buffer (number of entries over all episodes)
		self.episodes_count = 0		# Number of episodes currently in the replay buffer
		self.episodes = np.zeros((self.max_episode, 2))		# Keep track of episode start/finish indexes
		self.timesteps_count = 0		# Keeps track of the number of timesteps within an episode for sampling purposed
		# each of these are a stack
		# TODO: bug we will have to deal with later: we need to explicitly define datatypes when initting each new numpy array. otherwise annoying typecasting shenenagains

		self.state = [[]]
		self.action = [[]]
		self.next_state = [[]]
		self.reward = [[]]
		self.not_done = [[]]

		self.n_steps = n_steps
		self.batch_size = batch_size

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):

		"""
		Assume there's a numpy array
		"""

		# print("hi")
		#if self.episodes_count is 0:
		#	self.state.append(state)
		#	self.action.append(action)
		#	self.next_state.append(next_state)
		#	self.reward.append(reward)
		#	self.not_done.append(1. - done)
		#else:
		self.state[-1].append(state)
		self.action[-1].append(action)
		self.next_state[-1].append(next_state)
		self.reward[-1].append(reward)
		self.not_done[-1].append(1. - done)

		self.size += 1
		self.timesteps_count += 1
		#print("in ADD, timesteps_count: ",self.timesteps_count)

		# if episode has terminated
		if done:
			print("In add, episode is DONE")
			print("self.timesteps_count: ",self.timesteps_count)
			print("len(self.state): ",len(self.state))
			print("len(self.state[episodes_count]): ", len(self.state[self.episodes_count]))
			print("len(self.state[episodes_count][0]): ", len(self.state[self.episodes_count][0]))
			print("len(self.state[episodes_count][1]): ", len(self.state[self.episodes_count][1]))
			# increment episode count
			self.episodes_count += 1
			# init new set of numpy arrays
			# TODO: DEFINE DATA TYPES!!!
			# Append empty list to start new row for an episode
			self.state.append([])
			self.action.append([])
			self.next_state.append([])
			self.reward.append([])
			self.not_done.append([])

			# if over max num episodes
			if self.episodes_count > self.max_episode:
				print(".popleft() on all the stacks and subtract 1 from episodes count ")
				self.remove_episode()

	def remove_episode(self):
		"""
		Remove the oldest episode from the replay buffer (FIFO)
		"""
		self.state.pop(0)
		self.action.pop(0)
		self.next_state.pop(0)
		self.reward.pop(0)
		self.not_done.pop(0)

	def add_episode(self, start):
		"""
		Remove old
		Initialize np array
		"""
		# call it when each episode starts
		if start:
			print("in ADD EPISODE start: ",self.episodes_count)
			print("in ADD EPISODE start: self.max_episode", self.max_episode)
			print("in ADD EPISODE start: len(self.episodes)", len(self.episodes))
			print("in ADD EPISODE start, self.timesteps_count: ", self.timesteps_count)
			self.episodes[self.episodes_count, 0] = self.timesteps_count  # record the beginning index in the buffer (ptr)
		# call it when each episode ends
		else:
			print("in ADD EPISODE end: ", self.episodes_count-1)
			print("in ADD EPISODE end, self.timesteps_count: ",self.timesteps_count)
			self.episodes[self.episodes_count-1, 1] = self.timesteps_count  # record the ending index in the buffer (ptr)
			#self.episodes_count += 1
			print("len(self.episodes): ",len(self.episodes))
			self.timesteps_count = 0  # Reset count of timesteps within an episode
		return

	def sample(self):
		# deciding whether we grab expert or non expert trajectories.
		# depends on how many episodes we've added so far (has to be more than the threshold we set - 100 by default)
		print("IN STACK SAMPLE")
		#print("1) self.expert_episode_num: ", self.expert_episode_num)
		print("episodes_count: ", self.episodes_count)

		episode_idx = random.choice(np.arange(0, self.episodes_count)) # Choose one random episode between [0,episode_count)

		#print("episode_idx[0]: ", episode_idx[0])
		# note: episode is an array (with one element). so we need to access the element with `episode[0]`
		# right here, we're grabbing the RANGE of indices from the beginning index (held in the buffer) to the ending index of the trajectory held in the buffer
		# sample episode
		#ind = np.arange(self.episodes[episode[0], 0], self.episodes[episode[0], 1])

		# Get the beginning timestep index and the ending timestep index within an episode
		ind = np.arange(self.episodes[episode_idx, 0], self.episodes[episode_idx, 1])
		print("self.episodes[episode_idx, 0]: ", self.episodes[episode_idx, 0])
		print("self.episodes[episode_idx, 1]: ", self.episodes[episode_idx, 1])

		#print("episodes: ", self.episodes)
		#print("ind: ", ind.astype(int))
		#print("state: ", self.state[ind.astype(int)])

		print("len(self.state[0]): ", len(self.state[0]))
		print("***IN SAMPLE episode_idx: ",episode_idx)

		# if self.episodes_count > 10:
		#torch.FloatTensor(self.state[episode_idx][ind.astype(int)]).to(self.device),
		#torch.FloatTensor(self.action[episode_idx][ind.astype(int)]).to(self.device),
		#torch.FloatTensor(self.next_state[episode_idx][ind.astype(int)]).to(self.device),
		#torch.FloatTensor(self.reward[episode_idx][ind.astype(int)]).to(self.device),
		#torch.FloatTensor(self.not_done[episode_idx][ind.astype(int)]).to(self.device)
		#	pdb.set_trace()

		#print("in SAMPLE self.episodes_count: ",self.episodes_count)
		#ep_states = self.state[episode_idx[0]]
		#print("len(ep_states): ",len(ep_states))
		#sampled_state = [ep_states[i] for i in ind.astype(int)]
		#print("len(sampled_state): ", len(sampled_state))

		#sampled_action = np.array(self.action)[episode_idx]
		#sampled_next_state = np.array(self.next_state)[episode_idx]
		#sampled_reward = np.array(self.reward)[episode_idx]
		#sampled_done = np.array(self.not_done)[episode_idx]

		return (
			torch.FloatTensor(self.state[episode_idx]).to(self.device),
			torch.FloatTensor(self.action[episode_idx]).to(self.device),
			torch.FloatTensor(self.next_state[episode_idx]).to(self.device),
			torch.FloatTensor(self.reward[episode_idx]).to(self.device),
			torch.FloatTensor(self.not_done[episode_idx]).to(self.device)
		)


# A buffer that stores and sample based on episodes that have different step size
class ReplayBuffer_NStep(object):
	def __init__(self, state_dim, action_dim, expert_episode_num, max_episode=10100, n_steps=1, batch_size=64):
		self.max_episode = max_episode
		self.max_size = max_episode * 500
		self.ptr = 0
		self.size = 0
		self.expert_episode = 0
		self.agent_episode = expert_episode_num  # this is 100 by default
		# self.episode_step = episode_step
		self.expert_episode_num = expert_episode_num  # this is 100 by default

		# self.oldest_episode_num = expert_episode_num

		self.episodes = np.zeros((self.max_episode,
								  2))  # keep track each episode index - inner array contains the start index (in ptr) and end index (also in ptr form)
		self.episodes_count = 0
		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, action_dim))
		self.next_state = np.zeros((self.max_size, state_dim))
		self.reward = np.zeros((self.max_size, 1))
		self.not_done = np.zeros((self.max_size, 1))

		self.n_steps = n_steps
		self.batch_size = batch_size

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		# print("hi")
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size

		# if self.ptr >= self.max_size:
		# 	self.ptr =
		self.size = min((self.size + 1), self.max_size)

	def add_episode(self, start):

		### TODO: check replay buffer size and delete replay buffer accordingly for batch

		# call it when each episode starts
		if start:
			# First check if there's at least ~200 time steps left
			# if self.ptr + 200 >= self.max_size:
			# 	self.ptr = self.episodes[self.oldest_episode_num, 0]  # begin the overwrite (this is gonna be mega buggy)
			# 	# Note: until we change up data structure, the sampling with this method places bias on stuffs
			#
			# 	# increment the oldest episode num
			# 	self.oldest_episode_num += 1

			self.episodes[self.episodes_count, 0] = int(self.ptr)  # record the beginning index in the buffer (ptr)
		# call it when each episode ends
		else:
			self.episodes[self.episodes_count, 1] = int(self.ptr)  # record the ending index in the buffer (ptr)
			self.episodes_count += 1
			if self.episodes_count >= self.max_episode:
				self.episodes_count = self.expert_episode_num  # just start at the beginning - we're deprecating this replay buffer anyways

	def sample(self):
		# deciding whether we grab expert or non expert trajectories.
		# depends on how many episodes we've added so far (has to be more than the threshold we set - 100 by default)
		if self.expert_episode_num == 0:
			expert_or_random = "agent"
		elif self.episodes_count > self.expert_episode_num:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[0.7, 0.3])
		else:
			expert_or_random = "expert"

		print("expert_or_random: ", expert_or_random)
		# pick the episode index based on whether its an expert or not
		if expert_or_random == "expert":
			episode = np.random.randint(0, self.expert_episode_num, size=1)
		else:
			print("self.expert_episode_num: ", self.expert_episode_num)
			print("episodes_count: ", self.episodes_count)
			episode = np.random.randint(self.expert_episode_num, self.episodes_count, size=1)

		# note: episode is an array (with one element). so we need to access the element with `episode[0]`

		# right here, we're grabbing the RANGE of indices from the beginning index (held in the buffer) to the ending index of the trajectory held in the buffer
		# sample episode
		ind = np.arange(self.episodes[episode[0], 0], self.episodes[episode[0], 1])

		# if self.episodes_count > 10:
		#	pdb.set_trace()
		return (
			torch.FloatTensor(self.state[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.action[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.next_state[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.reward[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.not_done[ind.astype(int)]).to(self.device)
		)

	def sample_one_nstep(self):
		# deciding whether we grab expert or non expert trajectories.
		# depends on how many episodes we've added so far (has to be more than the threshold we set - 100 by default)
		if self.expert_episode_num == 0:
			expert_or_random = "agent"
		elif self.episodes_count > self.expert_episode_num:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[0.7, 0.3])
		else:
			expert_or_random = "expert"

		# pick the episode index based on whether its an expert or not
		if expert_or_random == "expert":
			# episode = np.random.randint(0, self.expert_episode_num, size=1)
			episode = np.random.randint(0, self.expert_episode_num)
		else:
			episode = np.random.randint(self.expert_episode_num, self.episodes_count)
			# episode = np.random.randint(self.oldest_episode_num, self.episodes_count)

		# right here, we're grabbing the RANGE of indices from the beginning index (held in the buffer) to the ending index of the trajectory held in the buffer
		# sample episode
		start_index = self.episodes[episode, 0]
		end_index = self.episodes[episode, 1]

		sample_start_index = np.random.randint(start_index, end_index + 1 - self.n_steps)  # exclusive of last index.

		ind = np.arange(sample_start_index, sample_start_index + self.n_steps)  # exclusive of last index

		# if self.episodes_count > 10:
		#	pdb.set_trace()
		return (
			torch.FloatTensor(self.state[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.action[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.next_state[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.reward[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.not_done[ind.astype(int)]).to(self.device)
		)

	def sample_batch_nstep(self):
		"""
		should return in shape (batch, n step length, relevant transition item's shape)
		"""
		# print('DFJASKLFJASDFKLJFASDKLJFASDKLJFAS')
		# print("==============================================================SHAPE OF SELF.STATE INSIDE REPLAY BUFFER")
		# print(self.state.shape)
		# print(self.action.shape)


		# deciding whether we grab expert or non expert trajectories.
		# depends on how many episodes we've added so far (has to be more than the threshold we set - 100 by default)
		if self.expert_episode_num == 0:
			expert_or_random = "agent"
		elif self.episodes_count > self.expert_episode_num:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[0.7, 0.3])
		else:
			expert_or_random = "expert"

		sample_start_index = np.zeros(self.batch_size)
		for i in range(self.batch_size):

			# pick the episode index based on whether its an expert or not
			if expert_or_random == "expert":
				# episode = np.random.randint(0, self.expert_episode_num, size=1)
				episode = np.random.randint(0, self.expert_episode_num)
			else:
				episode = np.random.randint(self.expert_episode_num, self.episodes_count)

			# right here, we're grabbing the RANGE of indices from the beginning index (held in the buffer) to the ending index of the trajectory held in the buffer
			# sample episode
			start_index = self.episodes[episode, 0]
			end_index = self.episodes[episode, 1]

			# print("expert policy: ")
			# print(expert_or_random == "expert")
			#
			# print("====================start index")
			# print(start_index)
			# print(end_index)
			# print("----------------------------------------------end")

			sample_start_index[i] = np.random.randint(start_index, end_index + 1 - self.n_steps)  # exclusive of last index.

		# # pick the episode index based on whether its an expert or not
		# if expert_or_random == "expert":
		# 	# episode = np.random.randint(0, self.expert_episode_num, size=1)
		# 	episode = np.random.randint(0, self.expert_episode_num)
		# else:
		# 	episode = np.random.randint(self.expert_episode_num, self.episodes_count)
		#
		#
		# # right here, we're grabbing the RANGE of indices from the beginning index (held in the buffer) to the ending index of the trajectory held in the buffer
		# # sample episode
		# start_index = self.episodes[episode, 0]
		# end_index = self.episodes[episode, 1]
		#
		# sample_start_index = np.random.randint(start_index, end_index + 1 - self.n_steps, size=self.batch_size)  # exclusive of last index.

		# np.linspace(wow, wow + 4, 5).T.astype(int)
		# What's happening here:
		# linspace is basically np.arange but for more than one number. since it gives the ranges by each vector, we then transpose it so each individual array in the batch is it's own n step trajectory range
		# batch_idx = np.linspace(sample_start_index, sample_start_index + self.n_steps - 1, self.n_steps).T.astype(int)

		# print("==================================================testing lalalala")
		# print(sample_start_index.shape)
		batch_idx = np.linspace(sample_start_index, sample_start_index + self.n_steps - 1, self.n_steps).T.astype(int)
		batch_idx = batch_idx.astype(int)
		# print("===================================================testing lalalala")
		# print(batch_idx.shape)
		# print(batch_idx)

		states = []
		actions = []
		next_states = []
		rewards = []
		not_dones = []

		for idx_arr in batch_idx:
			states.append(self.state[idx_arr])
			actions.append(self.action[idx_arr])
			next_states.append(self.next_state[idx_arr])
			rewards.append(self.reward[idx_arr])
			not_dones.append(self.not_done[idx_arr])

		states = np.array(states)
		actions = np.array(actions)
		next_states = np.array(next_states)
		rewards = np.array(rewards)
		not_dones = np.array(not_dones)


		# if self.episodes_count > 10:
		#	pdb.set_trace()

		# print("==============================================================SHAPE OF SELF.STATE INSIDE REPLAY BUFFER")
		# print(self.state.shape)
		# print(self.action.shape)

		return (
			torch.FloatTensor(states).to(self.device),
			torch.FloatTensor(actions).to(self.device),
			torch.FloatTensor(next_states).to(self.device),
			torch.FloatTensor(rewards).to(self.device),
			torch.FloatTensor(not_dones).to(self.device)
		)

	def sample_batch_1step(self):
		"""
		should return in shape (batch, n step length, relevant transition item's shape)
		"""
		# print('DFJASKLFJASDFKLJFASDKLJFASDKLJFAS')
		# print("==============================================================SHAPE OF SELF.STATE INSIDE REPLAY BUFFER")
		# print(self.state.shape)
		# print(self.action.shape)


		# deciding whether we grab expert or non expert trajectories.
		# depends on how many episodes we've added so far (has to be more than the threshold we set - 100 by default)
		if self.expert_episode_num == 0:
			expert_or_random = "agent"
		elif self.episodes_count > self.expert_episode_num:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[0.7, 0.3])
		else:
			expert_or_random = "expert"

		if expert_or_random == "expert":
			# episode = np.random.randint(0, self.expert_episode_num)
			episode_range = (0, self.episodes[self.expert_episode_num, 1])

		else:
			# episode = np.random.randint(self.expert_episode_num, self.episodes_count)
			episode_range = (self.episodes[self.expert_episode_num, 1] + 1, self.max_size)

		episode_indices = np.random.randint(episode_range[0], episode_range[1], size=self.batch_size)

		states = self.state[episode_indices]
		actions = self.action[episode_indices]
		next_states = self.next_state[episode_indices]
		rewards = self.reward[episode_indices]
		not_dones = self.not_done[episode_indices]

		states = np.expand_dims(states, axis=1)
		actions = np.expand_dims(actions, axis=1)
		next_states = np.expand_dims(next_states, axis=1)
		rewards = np.expand_dims(rewards, axis=1)
		not_dones = np.expand_dims(not_dones, axis=1)

		# print("=============testing")
		# print(states.shape)
		# print(actions.shape)
		# print(next_states.shape)
		# print(rewards.shape)
		# print(not_dones.shape)
		# print("stop")

		return (
			torch.FloatTensor(states).to(self.device),
			torch.FloatTensor(actions).to(self.device),
			torch.FloatTensor(next_states).to(self.device),
			torch.FloatTensor(rewards).to(self.device),
			torch.FloatTensor(not_dones).to(self.device)
		)

# A buffer that stores and sample based on episodes that have different step size
class ReplayBuffer_VarStepsEpisode(object):
	def __init__(self, state_dim, action_dim, expert_episode_num, max_episode=10100):
		self.max_episode = max_episode
		self.max_size = max_episode * 500
		self.ptr = 0
		self.size = 0
		self.expert_episode = 0
		self.agent_episode = expert_episode_num  # this is 100 by default
		# self.episode_step = episode_step
		self.expert_episode_num = expert_episode_num  # this is 100 by default

		self.episodes = np.zeros((self.max_episode, 2))  # keep track each episode index - inner array contains the start index (in ptr) and end index (also in ptr form)
		self.episodes_count = 0
		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, action_dim))
		self.next_state = np.zeros((self.max_size, state_dim))
		self.reward = np.zeros((self.max_size, 1))
		self.not_done = np.zeros((self.max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr += 1
		self.size += 1
		# pdb.set_trace()

	def add_episode(self, start):
		# call it when each episode starts
		if start:
			self.episodes[self.episodes_count, 0] = int(self.ptr)  # record the beginning index in the buffer (ptr)
		# call it when each episode ends
		else:
			self.episodes[self.episodes_count, 1] = int(self.ptr)  # record the ending index in the buffer (ptr)
			self.episodes_count += 1

	def sample(self):
		# deciding whether we grab expert or non expert trajectories.
		# depends on how many episodes we've added so far (has to be more than the threshold we set - 100 by default)
		print("IN SAMPLE")
		print("1) self.expert_episode_num: ", self.expert_episode_num)
		print("episodes_count: ",self.episodes_count)
		if self.expert_episode_num == 0:
			expert_or_random = "agent"
		elif self.episodes_count > self.expert_episode_num:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p = [0.7, 0.3])
		else:
			expert_or_random = "expert"

		print("expert_or_random: ",expert_or_random)
		# pick the episode index based on whether its an expert or not
		if expert_or_random == "expert":
			episode = np.random.randint(0, self.expert_episode_num, size = 1)
		else:
			print("self.expert_episode_num: ", self.expert_episode_num)
			print("episodes_count: ",self.episodes_count)
			episode = np.random.randint(self.expert_episode_num, self.episodes_count, size = 1)

		#note: episode is an array (with one element). so we need to access the element with `episode[0]`

		# right here, we're grabbing the RANGE of indices from the beginning index (held in the buffer) to the ending index of the trajectory held in the buffer
		# sample episode 
		ind = np.arange(self.episodes[episode[0], 0], self.episodes[episode[0], 1])
		print("self.episodes[episode[0], 0]: ", self.episodes[episode[0], 0])
		print("self.episodes[episode[0], 1]: ", self.episodes[episode[0], 1])

		print("episodes: ", self.episodes)
		print("ind: ",ind.astype(int))
		print("state: ", self.state[ind.astype(int)])
		
		#if self.episodes_count > 10:
		#	pdb.set_trace()
		return (
			torch.FloatTensor(self.state[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.action[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.next_state[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.reward[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.not_done[ind.astype(int)]).to(self.device)
		)


"""
Yi's old replay buffer
"""
class ReplayBuffer_episode(object):
	def __init__(self, state_dim, action_dim, episode_step, expert_episode_num, max_episode=10100):
		self.max_episode = max_episode
		self.max_size = max_episode * episode_step
		self.ptr = 0
		self.size = 0
		self.expert_episode = 0
		self.agent_episode = expert_episode_num
		self.episode_step = episode_step
		self.expert_episode_num = expert_episode_num
		self.episode = 0

		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, action_dim))
		self.next_state = np.zeros((self.max_size, state_dim))
		self.reward = np.zeros((self.max_size, 1))
		self.not_done = np.zeros((self.max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size

		self.size = min(self.size + 1, self.max_size)

		if self.size % self.episode_step == 0 and self.size <= self.episode_step * self.expert_episode_num:
			self.expert_episode = self.expert_episode + 1
		elif self.size % self.episode_step == 0 and self.size > self.episode_step * self.expert_episode_num:
			# pdb.set_trace()
			self.agent_episode = self.agent_episode + 1

	def sample(self):
		# ind = np.random.randint(0, self.size, size=batch_size)
		if self.agent_episode > self.expert_episode_num:
			# pdb.set_trace()
			prob = np.random.choice(np.array(["expert", "agent"]), p = [0.7, 0.3])
		else:
			prob = "expert"

		# sample expert episode
		if prob	== "expert" and self.expert_episode > 0:
			random_episode = np.random.randint(1, self.expert_episode + 1, size=1)
		else:
		# sample agent episode
			random_episode = np.random.randint(self.expert_episode + 1, self.agent_episode + 1, size = 1)

		# sample episode
		ind = np.arange((random_episode[0] - 1)*self.episode_step, random_episode[0]*self.episode_step)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def add_wo_expert(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size

		self.size = min(self.size + 1, self.max_size)

		if self.size % self.episode_step == 0:
			self.episode += 1

	def sample_wo_expert(self):
		# sample episode
		random_episode = np.random.randint(0, self.episode, size=1)
		ind = np.arange(random_episode[0]*self.episode_step, (random_episode[0]*self.episode_step) + 100)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class ReplayBuffer_random(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)