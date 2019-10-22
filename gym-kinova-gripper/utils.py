import numpy as np
import torch
import pdb

class ReplayBuffer_episode(object):
	def __init__(self, state_dim, action_dim, episode_step, expert_episode_num, max_episode=10000):
		self.max_episode = max_episode
		self.max_size = max_episode * episode_step
		self.ptr = 0
		self.size = 0
		self.expert_episode = 0
		self.agent_episode = 100
		self.episode_step = episode_step
		self.expert_episode_num = expert_episode_num

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
			self.agent_episode = self.agent_episode + 1

	def sample(self):
		# ind = np.random.randint(0, self.size, size=batch_size)
		if self.agent_episode > self.expert_episode:
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
		# pdb.set_trace()

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