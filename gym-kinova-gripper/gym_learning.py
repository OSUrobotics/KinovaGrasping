#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:19:36 2020

@author: orochi
"""
import gym
import DDPG
import torch
import wierd
import numpy as np
import utils


env = gym.make('LunarLander-v2')
env.reset()
env.env.continuous=True
while True:
    env.step(np.random.rand(2))
    env.render()
print('should be rendering')
env.close()
'''
train_prev_state=[]
train_action=[]
train_observation=[]
train_reward=[]
if type(env.action_space)==gym.spaces.discrete.Discrete:
    max_action=1
    action_dim=1#env.action_space.n
else:
    max_action=env.action_space.high
    action_dim=len(env.action_space.high)
if type(env.observation_space)==gym.spaces.discrete.Discrete:
    state_dim=env.observation_space.n
else:
    state_dim=len(env.observation_space.high)
policy=wierd.DDPG(state_dim, action_dim, max_action, discount=0.99, tau=0.0005)
print('new policy made')
for i_episode in range(2000):
    prev_state = env.reset()
    for t in range(100):
        #env.render()
        action=policy.select_action(prev_state)
        observation, reward, done, info = env.step(int(action.item()))
        #print('reward done',reward,done)
        critic_loss=policy.train_sample(prev_state,action, observation,reward, not(done), batch_size=64)
        prev_state=observation
        print('critic loss',critic_loss.item())

        #print(critic_loss, actor_loss)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
'''