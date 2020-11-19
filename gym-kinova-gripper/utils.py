import numpy as np
import torch
import pdb


# A buffer that stores and sample based on episodes that have different step size
class ReplayBuffer_NStep(object):
	def __init__(self, state_dim, action_dim, expert_episode_num, max_episode=10100, n_steps=5, batch_size=64):
		self.max_episode = max_episode
		self.max_size = max_episode * 500
		self.ptr = 0
		self.size = 0
		self.expert_episode = 0
		self.agent_episode = expert_episode_num  # this is 100 by default
		# self.episode_step = episode_step
		self.expert_episode_num = expert_episode_num  # this is 100 by default

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
		if self.episodes_count > self.expert_episode_num:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[0.7, 0.3])
		else:
			expert_or_random = "expert"

		# pick the episode index based on whether its an expert or not
		if expert_or_random == "expert":
			episode = np.random.randint(0, self.expert_episode_num, size=1)
		else:
			episode = np.random.randint(self.expert_episode_num, self.episodes_count, size=1)

		# note: episode is an array (with one element). so we need to access the element with `episode[0]`

		# right here, we're grabbing the RANGE of indices from the beginning index (held in the buffer) to the ending index of the trajectory held in the buffer
		# sample episode
		ind = np.arange(self.episodes[episode[0], 0], self.episodes[episode[0], 1] + 1)

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
		if self.episodes_count > self.expert_episode_num:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[0.7, 0.3])
		else:
			expert_or_random = "expert"

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
		if self.episodes_count > self.expert_episode_num:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[0.7, 0.3])
		else:
			expert_or_random = "expert"

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

		sample_start_index = np.random.randint(start_index, end_index + 1 - self.n_steps, size=self.batch_size)  # exclusive of last index.

		# np.linspace(wow, wow + 4, 5).T.astype(int)
		# What's happening here:
		# linspace is basically np.arange but for more than one number. since it gives the ranges by each vector, we then transpose it so each individual array in the batch is it's own n step trajectory range
		batch_idx = np.linspace(sample_start_index, sample_start_index + self.n_steps - 1, self.n_steps).T.astype(int)
		batch_idx = batch_idx.astype(int)

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

		print("==============================================================SHAPE OF SELF.STATE INSIDE REPLAY BUFFER")
		print(self.state.shape)
		print(self.action.shape)

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