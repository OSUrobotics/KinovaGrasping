# from envs.kinova_gripper_env import KinovaGripper_Env
# env.action_space.sample()
import gym
from gym import spaces
import numpy as np
import pdb
env = gym.make('gym_kinova_gripper:kinovagripper-v0')

env.reset()

finger = np.array([0.5, 0.5, 0.5])
# finger = np.array([0.0, 0.4, 0.4, 0.5])
reward_total = 0
# print(env.action_space)
step = 0
# print(env._max_episode_steps)
for i in range(50):
	obs, reward, done, _ = env.step(finger)
	# # env.render()
	# if i == 50:
	# 	finger = np.array([-0.2, 0.0, 0.0])
	# if i == 100:
	# 	finger = np.array([0.2, 0.0, 0.0])
	# pdb.set_trace()
	# reward_total += reward 
	print("reward", reward)
	# print("dot_prod", dot_prod)

	# if abs(dot_prod) > 0.9:
	# if step == 50:

	# 	# print("here")
	# 	finger = np.array([0.3, 0.8, 0.8, 0.8])
	# step += 1	
	# print("obs", len(obs))
	# print("done", done)
	# print(type(env._sim.data.time))
	# print(env.action_space.sample())
	# if abs(env._sim.data.time - 2.000) < 0.0000001:
	# 	print(env._sim.data.get_joint_qpos("j2s7s300_joint_finger_1") / 2) 
	# print(done)

print(env.env._sim.data.time)
# print(env._elapsed_steps)

# obs_min = np.array([-0.1, -0.1, 0.0, -360, -360, -360, -0.1, -0.1, 0.0, -360, -360, -360,
# 	-0.1, -0.1, 0.0, -360, -360, -360,-0.1, -0.1, 0.0, -360, -360, -360,
# 	-0.1, -0.1, 0.0, -360, -360, -360,-0.1, -0.1, 0.0, -360, -360, -360, 
# 	-0.1, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# obs_max = np.array([0.1, 0.1, 0.3, 360, 360, 360, 0.1, 0.1, 0.3, 360, 360, 360,
# 	0.1, 0.1, 0.3, 360, 360, 360,0.1, 0.1, 0.3, 360, 360, 360,
# 	0.1, 0.1, 0.3, 360, 360, 360,0.1, 0.1, 0.3, 360, 360, 360,
# 	0.1, 0.7, 0.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

# # print(len(obs_max))
# # obs_min = np.zeros(17)
# # obs_max = obs_min + np.Inf
# # print(type(np.Inf))
# a = spaces.Box(low=obs_min, high=obs_max, dtype=np.float32)
# b = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
# print(b.shape)


# obs_min = np.zeros(17) 
# obs_max = obs_min + np.Inf
# c = spaces.Box(low=obs_min , high=obs_max, dtype=np.float32)
# print(c.shape)