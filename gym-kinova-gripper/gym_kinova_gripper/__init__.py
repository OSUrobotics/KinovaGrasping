from gym.envs.registration import register

register(
    id='kinovagripper-v0',
    entry_point='gym_kinova_gripper.envs:KinovaGripper_Env',
    max_episode_steps = 45
)