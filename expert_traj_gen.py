#!/usr/bin/env python3

###############
# Author: Yi Herng Ong
# Purpose: collect several strategies in advance from PID controlller from human demonstration and pretrain network
# Summer 2019

###############

import gym


class ExpertTrajGen():
	def __init__(self, obs, strategy_index):
		self.obs = obs
		self.all_strategies = ["nudge", "preshape"]
		self.chosen_strategy = self.all_strategies[strategy_index]


	def _generate(self):
		