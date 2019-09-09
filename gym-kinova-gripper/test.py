# from envs.kinova_gripper_env import KinovaGripper_Env
# env.action_space.sample()
import gym
import numpy as np
env = gym.make('gym_kinova_gripper:kinovagripper-v0')

env.reset()

for _ in range(5):
	obs, reward, done, _ = env.step(np.array([[1.0, 0.0, 0.0]]))
	# env.render()
	print("reward", reward)
	print("done", done)