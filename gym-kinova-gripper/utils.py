import numpy as np
import torch
import pdb
import random
import os
import datetime
from pathlib import Path

class ReplayBuffer_Queue(object):
	def __init__(self, state_dim, action_dim, max_episode=10000, n_steps=5):
		self.max_episode = max_episode		# Maximum number of episodes, limit to when we remove old episodes
		self.size = 0				# Full size of the replay buffer (number of entries over all episodes)
		self.episodes_count = 0		# Number of episodes that have occurred (may be more than max replay buffer side)
		self.replay_ep_num = 0		# Number of episodes currently in the replay buffer
		self.episodes = [[]]		# Keep track of episode start/finish indexes
		self.timesteps_count = 0		# Keeps track of the number of timesteps within an episode for sampling purposed

		# each of these are a queue
		self.state = [[]]
		self.action = [[]]
		self.next_state = [[]]
		self.reward = [[]]
		self.not_done = [[]]

		self.finger_reward = [[]]
		self.grasp_reward = [[]]
		self.lift_reward = [[]]
		self.n_steps = n_steps

		self.orientation_indexes = [] # Hand pose variation and noise

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):

		"""
		Assume there's a numpy array
		"""
		self.state[-1].append(state)
		self.action[-1].append(action)
		self.next_state[-1].append(next_state)
		self.reward[-1].append(reward)
		self.not_done[-1].append(1. - done)

		self.size += 1
		self.timesteps_count += 1

	def remove_episode(self, idx=0):
		"""
		Remove the oldest episode from the replay buffer (FIFO)
		"""
		self.state.pop(idx)
		self.action.pop(idx)
		self.next_state.pop(idx)
		self.reward.pop(idx)
		self.not_done.pop(idx)
		self.episodes.pop(idx)
		if len(self.orientation_indexes) > 0:
			self.orientation_indexes.pop(idx)

		self.replay_ep_num -= 1

	def add_episode(self, start):
		"""
		Remove old
		Initialize np array
		"""
		# call it when each episode starts
		if start:
			self.episodes[-1].append(self.timesteps_count)
		# call it when each episode ends
		else:
			self.episodes[-1].append(self.timesteps_count)
			self.episodes.append([])
			self.timesteps_count = 0  # Reset count of timesteps within an episode
		return

	def add_orientation_idx_to_replay(self, idx):
		""" Add orientation noise for each episode """
		# Appends for each episode
		self.orientation_indexes.append(idx)

	def sample(self):
		""" Sample one episode from replay buffer, learn from full trajectory """
		# Choose one random episode between [0,episode_count)
		episode_idx = random.choice(np.arange(0, self.replay_ep_num))

		# Get the beginning timestep index and the ending timestep index within an episode
		selected_indexes = np.arange(self.episodes[episode_idx][0], self.episodes[episode_idx][1])

		return (
			torch.FloatTensor([self.state[episode_idx][x] for x in selected_indexes]).to(self.device),
			torch.FloatTensor([self.action[episode_idx][x] for x in selected_indexes]).to(self.device),
			torch.FloatTensor([self.next_state[episode_idx][x] for x in selected_indexes]).to(self.device),
			torch.FloatTensor([self.reward[episode_idx][x] for x in selected_indexes]).to(self.device),
			torch.FloatTensor([self.not_done[episode_idx][x] for x in selected_indexes]).to(self.device)
		)

	def sample_batch_nstep(self,batch_size):
		""" Samples batch size of replay buffer trajectories for learning using n-step returns """
		# Initialize arrays
		state_arr = []
		action_arr = []
		next_state_arr = []
		reward_arr = []
		not_done_arr = []

		if batch_size == 0 or self.replay_ep_num == 0 or self.replay_ep_num < batch_size:
			return (
				torch.FloatTensor(state_arr).to(self.device),
				torch.FloatTensor(action_arr).to(self.device),
				torch.FloatTensor(next_state_arr).to(self.device),
				torch.FloatTensor(reward_arr).to(self.device),
				torch.FloatTensor(not_done_arr).to(self.device)
			)

		# List of randomly-selected episode indices based on current number of episodes
		episode_idx_arr = np.random.randint(max(1,self.replay_ep_num - 1), size=batch_size)

		for idx in episode_idx_arr:
			# Get episode length (number of time steps)
			episode_len = len(self.state[idx])

			# get the ceiling idx. note the stagger b/c of n steps. the 1 is so that we don't pick 0 as an index (see next part)
			ceiling = episode_len - self.n_steps

			# Get the trajectory from starting index to n_steps later
			trajectory_arr_idx = []

			for num in range(ceiling-1):
				start_idx = num #np.random.randint(ceiling)
				trajectory_arr_idx.append(np.arange(start_idx, start_idx + self.n_steps))

			trajectory_arr_idx.append(np.arange(ceiling, ceiling + self.n_steps))

			temp_state = np.array(self.state[idx])
			temp_action = np.array(self.action[idx])
			temp_next_state = np.array(self.next_state[idx])
			temp_reward = np.array(self.reward[idx])
			temp_not_done = np.array(self.not_done[idx])

			for row in trajectory_arr_idx:
				state_arr.append(temp_state[row])
				action_arr.append(temp_action[row])
				next_state_arr.append(temp_next_state[row])
				reward_arr.append(temp_reward[row])
				not_done_arr.append(temp_not_done[row])

		return (
			torch.FloatTensor(state_arr).to(self.device),
			torch.FloatTensor(action_arr).to(self.device),
			torch.FloatTensor(next_state_arr).to(self.device),
			torch.FloatTensor(reward_arr).to(self.device),
			torch.FloatTensor(not_done_arr).to(self.device)
		)


	def replace(self, reward, done):
		"""
		Used to replace the last time step of an episode
		to include lift reward and set the done bit as True
		@param reward: The updated lift reward value
		@param done: The updated done bit value
		"""

		# Current episode index
		episode_idx = self.replay_ep_num - 1
		if len(self.reward[episode_idx]) == 0:
			print("We cannot replace the reward as for this episode as we have not done any transitions!")
			return 0

		if not done:
			print("Can only replace last time step of the latest episode")
			raise ValueError

		idx = len(self.reward[episode_idx]) - 1
		old_reward = self.reward[episode_idx][idx]
		self.reward[episode_idx][idx] = reward
		self.not_done[episode_idx][idx] = 1. - float(done)

		# increment episode count
		self.episodes_count += 1
		self.replay_ep_num += 1
		# print("ADDED DONE BIT")

		# Append empty list to start new row for an episode
		self.state.append([])
		self.action.append([])
		self.next_state.append([])
		self.reward.append([])
		self.not_done.append([])

		# If over max number of episodes for replay buffer
		if self.replay_ep_num > self.max_episode:
			self.remove_episode()

		return old_reward

	def save_replay_buffer(self,save_filepath):
		""" Save replay buffer to saving directory """
		save_path = Path(save_filepath)
		save_path.mkdir(parents=True, exist_ok=True)
		save_filepath += "/"

		print("Saving replay buffer...")
		np.save(save_filepath + "state", self.state)
		np.save(save_filepath + "action", self.action)
		np.save(save_filepath + "next_state", self.next_state)
		np.save(save_filepath + "reward", self.reward)
		np.save(save_filepath + "not_done", self.not_done)

		np.save(save_filepath + "episodes", self.episodes)  # Keep track of episode start/finish indexes
		np.save(save_filepath + "episodes_info",
				[self.max_episode, self.size, self.episodes_count,
				 self.replay_ep_num])
		np.save(save_filepath + "orientation_indexes", self.orientation_indexes) # Keep track of orientation noise at each timestep
		# max_episode: Maximum number of episodes, limit to when we remove old episodes
		# size: Full size of the replay buffer (number of entries over all episodes)
		# episodes_count: Number of episodes that have occurred (may be more than max replay buffer side)
		# replay_ep_num: Number of episodes currently in the replay buffer
		return save_filepath

	def evaluate_replay_reward(self,reward):
		""" Check the contents of the reward values stored within the replay buffer
		replay_buffer: replay buffer to evaluate
		filepath: filepath where replay buffer is stored (for recording purposes)
		returns: Text to describe the reward contents (recorded in the info.txt file)
		"""
		non_zero_count = 0
		finger_reward_count = 0
		grasp_reward_count = 0
		lift_reward_count = 0
		num_rows = 0
		num_elems = 0
		i = 0

		in_check = 0

		for row in reward:
			num_rows += 1
			for elem in row:
				num_elems += 1
				if elem != 0:
					non_zero_count += 1
					if elem < 5:
						finger_reward_count += 1
					elif elem < 10:
						grasp_reward_count += 1
					elif elem > 49:
						lift_reward_count += 1

			#print("in_check: ",in_check)

			i += 1
		#episode_idx_arr = random.sample(success_idx, 200) # Just successful indexes

		text = "\nnon_zero_reward: "+str(non_zero_count)
		text += "\nfinger_reward_count: "+str(finger_reward_count)
		text += "\ngrasp_reward_count: "+str(grasp_reward_count)
		text += "\nlift_reward_count: "+str(lift_reward_count)
		text += "\nnum_rows: "+str(num_rows)

		return text, num_elems


	def store_saved_data_into_replay(self, filepath, sample_size=None):
		""" Restore replay buffer from saved location
		filepath: Filepath of the stored replay buffer
		sample_size: Number of episodes to get from the end of the replay buffer (most recent experience)
		"""
		if filepath is None or os.path.isdir(filepath) is False:
			print("Replay buffer not found!! filepath: ", filepath)
			quit()

		print("In store_saved_data_into_replay, filepath: ",filepath)

		expert_state = np.load(filepath + "state.npy", allow_pickle=True).astype('object')
		expert_action = np.load(filepath + "action.npy", allow_pickle=True).astype('object')
		expert_next_state = np.load(filepath + "next_state.npy", allow_pickle=True).astype('object')
		expert_reward = np.load(filepath + "reward.npy", allow_pickle=True).astype('object')
		expert_not_done = np.load(filepath + "not_done.npy", allow_pickle=True).astype('object')

		expert_episodes = np.load(filepath + "episodes.npy", allow_pickle=True).astype(
			'object')  # Keep track of episode start/finish indexes
		expert_episodes_info = np.load(filepath + "episodes_info.npy", allow_pickle=True)
		# Hand pose orientation per time step per episode
		orient_file = Path(filepath + "orientation_indexes.npy")
		if orient_file.is_file():
			expert_orientation_indexes = np.load(filepath + "orientation_indexes.npy", allow_pickle=True).astype('object')
			self.orientation_indexes = expert_orientation_indexes.tolist()

		if sample_size is not None:
			expert_state = expert_state[-sample_size:]
			expert_action = expert_action[-sample_size:]
			expert_next_state = expert_next_state[-sample_size:]
			expert_reward = expert_reward[-sample_size:]
			expert_not_done = expert_not_done[-sample_size:]
			expert_episodes = expert_episodes[-sample_size:]
			expert_episodes_info = expert_episodes_info[-sample_size:]
			expert_orientation_indexes = expert_orientation_indexes[-sample_size:]

		# If first replay buffer is being added, set to exact values
		if self.size == 0:
			self.state = expert_state.tolist()
			self.action = expert_action.tolist()
			self.next_state = expert_next_state.tolist()
			self.reward = expert_reward.tolist()
			self.not_done = expert_not_done.tolist()
			self.episodes = expert_episodes.tolist()
		else:
			# Otherwise, extend current replay buffer with new replay data
			# Convert numpy array to list and set to replay buffer
			self.state.extend(expert_state.tolist())
			self.action.extend(expert_action.tolist())
			self.next_state.extend(expert_next_state.tolist())
			self.reward.extend(expert_reward.tolist())
			self.not_done.extend(expert_not_done.tolist())
			self.episodes.extend(expert_episodes.tolist())

		# Print out the values stored within the reward array of the replay buffer
		reward_text, num_elems = self.evaluate_replay_reward(self.reward)

		if sample_size is not None:
			self.max_episode += sample_size
			self.size += num_elems
			self.episodes_count += sample_size
			self.replay_ep_num += sample_size
		else:
			self.max_episode += expert_episodes_info[0]
			self.size += expert_episodes_info[1]
			self.episodes_count += expert_episodes_info[2]
			self.replay_ep_num += expert_episodes_info[3]

		# max_episode: Maximum number of episodes, limit to when we remove old episodes
		# size: Full size of the replay buffer (number of entries over all episodes)
		# episodes_count: Number of episodes that have occurred (may be more than max replay buffer side)
		# replay_ep_num: Number of episodes currently in the replay buffer

		text = ""
		text += "\nREPLAY BUFFER: "
		text += "\nfilepath: " + filepath
		text += "\nreplay_buffer.replay_ep_num: "+str(self.replay_ep_num)
		text += "\nreplay_buffer.size: "+str(self.size)
		text += "\nreplay_buffer.timesteps_count: "+str(self.timesteps_count)
		text += reward_text

		print(text)

		# If there is a trailing empty entry [], remove it
		if len(self.reward[-1]) == 0:
			self.remove_episode(-1)

		return text


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
		self.size = (self.size + 1) % self.max_size

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
			episode = np.random.randint(self.expert_episode_num, self.episodes_count, size = 1)

		#note: episode is an array (with one element). so we need to access the element with `episode[0]`

		# right here, we're grabbing the RANGE of indices from the beginning index (held in the buffer) to the ending index of the trajectory held in the buffer
		# sample episode
		ind = np.arange(self.episodes[episode[0], 0], self.episodes[episode[0], 1])

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
