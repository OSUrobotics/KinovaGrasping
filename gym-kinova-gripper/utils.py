import numpy as np
import torch
import pdb

# A buffer that stores and sample based on episodes that have different step size
class ReplayBuffer_VarStepsEpisode(object):
	def __init__(self, state_dim, action_dim, expert_episode_num, max_episode=10100):
		self.max_episode = max_episode
		self.max_size = max_episode * 500
		self.ptr = 0
		self.size = 0
		self.expert_episode = 0
		self.agent_episode = expert_episode_num
		# self.episode_step = episode_step
		self.expert_episode_num = expert_episode_num

		self.episodes = np.zeros((self.max_episode, 2)) # keep track each episode index
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
			self.episodes[self.episodes_count, 0] = int(self.ptr)
		# call it when each episode ends
		else:
			self.episodes[self.episodes_count, 1] = int(self.ptr)			
			self.episodes_count += 1

	def sample(self):
		if self.episodes_count > self.expert_episode_num:
			expert_or_random = np.random.choice(np.array(["expert", "agent"]), p = [0.7, 0.3])
		else:
			expert_or_random = "expert"
		
		if expert_or_random == "expert":
			episode = np.random.randint(0, self.expert_episode_num, size = 1)
		else:
			episode = np.random.randint(self.expert_episode_num, self.episodes_count, size = 1)
		
		# sample episode 
		ind = np.arange(self.episodes[episode[0], 0], self.episodes[episode[0], 1] + 1)
		
		if self.episodes_count > 10:
			pdb.set_trace()
		return (
			torch.FloatTensor(self.state[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.action[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.next_state[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.reward[ind.astype(int)]).to(self.device),
			torch.FloatTensor(self.not_done[ind.astype(int)]).to(self.device)
		)		


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
		if prob	== "expert":
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